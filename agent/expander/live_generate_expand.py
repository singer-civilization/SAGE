import json
import time
import traceback
from collections import deque
from dataclasses import dataclass

from RL.gen_slow_sql import SeedGeneratorSession
from agent.expander.config import ExpanderSettings, load_settings
from agent.expander.io_utils import log_line, reset_run_files
from agent.expander.validation import structural_fingerprint
from agent.expander.workflow import (
    ExpanderRuntimeState,
    SeedState,
    build_runtime_state,
    close_runtime_state,
    expand_one_seed,
)
from agent.planner import build_plan_bundle
from agent.planner.models import PathConstraint
from agent.planner.workflow import PlanBundle
from agent.planner.workflow import save_plan_outputs


@dataclass
class PathRuntimeState:

    path_constraint: PathConstraint
    status: str = "ready"
    accept_limit: int = 0
    generated_seed_count: int = 0
    accepted_sql_count: int = 0
    failed_seed_count: int = 0
    no_gain_turns: int = 0
    turn_count: int = 0
    last_selected_turn: int = 0
    last_gain_turn: int = 0
    session: SeedGeneratorSession | None = None


def build_seed_session(
    settings: ExpanderSettings,
    path_constraint: PathConstraint,
    summary: dict,
    shared_agent,
    *,
    emit_output: bool = False,
) -> SeedGeneratorSession:

    seed_settings = settings.seed_generator
    return SeedGeneratorSession(
        dbname=seed_settings.dbname,
        target=seed_settings.target_seconds,
        accept_min_ratio=seed_settings.accept_min_ratio,
        accept_max_ratio=seed_settings.accept_max_ratio,
        timeout_ratio=seed_settings.timeout_ratio,
        from_mode=path_constraint.from_mode,
        max_steps=seed_settings.max_steps,
        emit_output=emit_output,
        write_output=True,
        path_constraint=path_constraint,
        summary=summary,
        shared_agent=shared_agent,
    )


def summarize_path_stats(runtime_state: ExpanderRuntimeState) -> list[dict]:

    summaries = []
    for path_id, stats in runtime_state.path_stats.items():
        summaries.append({"path_id": path_id, **stats})
    return summaries


def path_constraint_from_dict(payload: dict) -> PathConstraint:

    return PathConstraint(
        path_id=payload["path_id"],
        path_name=payload["path_name"],
        allowed_tables=tuple(payload["allowed_tables"]),
        allowed_join_paths=tuple(tuple(edge) for edge in payload["allowed_join_paths"]),
        primary_tables=tuple(payload["primary_tables"]),
        secondary_tables=tuple(payload["secondary_tables"]),
        from_mode=payload["from_mode"],
        required_predicate=bool(payload["required_predicate"]),
        allowed_clauses=tuple(payload["allowed_clauses"]),
        preferred_predicate_columns=tuple(payload["preferred_predicate_columns"]),
        preferred_group_columns=tuple(payload["preferred_group_columns"]),
        preferred_order_columns=tuple(payload["preferred_order_columns"]),
        allow_aggregate=bool(payload["allow_aggregate"]),
        path_reason=payload["path_reason"],
        probe_sql=payload.get("probe_sql", ""),
        probe_runtime_seconds=float(payload.get("probe_runtime_seconds", -1.0)),
    )


def load_plan_bundle_from_checkpoint(settings: ExpanderSettings, summary: dict, checkpoint: dict) -> PlanBundle:

    if checkpoint.get("paths"):
        paths = tuple(path_constraint_from_dict(path) for path in checkpoint["paths"])
    elif settings.planner_constraints_file.exists():
        saved_paths = json.loads(settings.planner_constraints_file.read_text(encoding="utf-8"))
        paths = tuple(path_constraint_from_dict(path) for path in saved_paths)
    else:
        raise RuntimeError("Cannot resume: missing checkpoint paths and planner constraints file")
    bucket_counts = {"single": 0, "pair": 0, "chain": 0}
    for path in paths:
        if len(path.allowed_tables) == 1:
            bucket_counts["single"] += 1
        elif len(path.allowed_tables) == 2:
            bucket_counts["pair"] += 1
        else:
            bucket_counts["chain"] += 1
    return PlanBundle(
        summary=summary,
        paths=paths,
        candidate_count=int(checkpoint.get("candidate_count", len(paths))),
        candidate_bucket_counts=checkpoint.get("candidate_bucket_counts", bucket_counts),
    )


def load_checkpoint(settings: ExpanderSettings) -> dict | None:

    if not settings.resume_run or not settings.checkpoint_file.exists():
        return None
    raw_checkpoint = settings.checkpoint_file.read_text(encoding="utf-8").strip()
    if not raw_checkpoint:
        return None
    return json.loads(raw_checkpoint)


def save_checkpoint(
    settings: ExpanderSettings,
    plan_bundle: PlanBundle,
    path_states: list[PathRuntimeState],
    ready_queue,
    *,
    generated_seed_count: int,
    global_turn_index: int,
    round_index: int,
    accepted_count: int,
    status: str = "running",
) -> None:

    payload = {
        "status": status,
        "generated_seed_count": generated_seed_count,
        "global_turn_index": global_turn_index,
        "round_index": round_index,
        "accepted_count": accepted_count,
        "candidate_count": plan_bundle.candidate_count,
        "candidate_bucket_counts": plan_bundle.candidate_bucket_counts,
        "paths": [path.to_dict() for path in plan_bundle.paths],
        "ready_queue": list(ready_queue),
        "path_states": [
            {
                "status": path_state.status,
                "accept_limit": path_state.accept_limit,
                "generated_seed_count": path_state.generated_seed_count,
                "accepted_sql_count": path_state.accepted_sql_count,
                "failed_seed_count": path_state.failed_seed_count,
                "no_gain_turns": path_state.no_gain_turns,
                "turn_count": path_state.turn_count,
                "last_selected_turn": path_state.last_selected_turn,
                "last_gain_turn": path_state.last_gain_turn,
            }
            for path_state in path_states
        ],
    }
    settings.checkpoint_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_path_states(
    checkpoint: dict,
    plan_bundle: PlanBundle,
    default_accept_limit: int,
) -> list[PathRuntimeState]:

    saved_states = checkpoint.get("path_states", [])
    restored_states: list[PathRuntimeState] = []
    for index, path_constraint in enumerate(plan_bundle.paths):
        saved_state = saved_states[index] if index < len(saved_states) else {}
        restored_states.append(
            PathRuntimeState(
                path_constraint=path_constraint,
                status=saved_state.get("status", "ready"),
                accept_limit=int(saved_state.get("accept_limit", default_accept_limit)),
                generated_seed_count=int(saved_state.get("generated_seed_count", 0)),
                accepted_sql_count=int(saved_state.get("accepted_sql_count", 0)),
                failed_seed_count=int(saved_state.get("failed_seed_count", 0)),
                no_gain_turns=int(saved_state.get("no_gain_turns", 0)),
                turn_count=int(saved_state.get("turn_count", 0)),
                last_selected_turn=int(saved_state.get("last_selected_turn", 0)),
                last_gain_turn=int(saved_state.get("last_gain_turn", 0)),
            )
        )
    return restored_states


def ensure_session(
    settings: ExpanderSettings,
    path_state: PathRuntimeState,
    summary: dict,
    shared_agent,
    *,
    emit_output: bool = False,
):

    if path_state.session is None:
        path_state.session = build_seed_session(
            settings,
            path_state.path_constraint,
            summary,
            shared_agent,
            emit_output=emit_output,
        )
    return path_state.session


def run_generator_turn(
    settings: ExpanderSettings,
    runtime_state: ExpanderRuntimeState,
    path_state: PathRuntimeState,
    generated_seed_index: int,
    summary: dict,
    shared_agent,
    *,
    emit_output: bool = False,
) -> tuple[bool, int, object]:

    session_started_at = time.monotonic()
    log_line(
        settings.log_file,
        f"path_id={path_state.path_constraint.path_id} generator_session_prepare "
        f"generated_seed_index={generated_seed_index}",
    )
    session = ensure_session(settings, path_state, summary, shared_agent, emit_output=emit_output)
    log_line(
        settings.log_file,
        f"path_id={path_state.path_constraint.path_id} generator_session_ready "
        f"generated_seed_index={generated_seed_index} elapsed={time.monotonic() - session_started_at:.3f}s",
    )
    generator_started_at = time.monotonic()
    log_line(
        settings.log_file,
        f"path_id={path_state.path_constraint.path_id} generator_start "
        f"generated_seed_index={generated_seed_index} "
        f"turn_attempt_limit={settings.live_generator_attempts_per_turn}",
    )
    seed_record = session.generate_next_seed(max_attempts=settings.live_generator_attempts_per_turn)
    generator_elapsed = time.monotonic() - generator_started_at
    if seed_record is None:
        path_state.failed_seed_count += 1
        log_line(
            settings.log_file,
            f"path_id={path_state.path_constraint.path_id} generator_exhausted "
            f"turn_attempt_limit={settings.live_generator_attempts_per_turn} "
            f"elapsed={generator_elapsed:.3f}s",
        )
        return False, 0, session.agent

    path_state.generated_seed_count += 1
    seed_sql = seed_record["sql"]
    runtime_state.attempted_fingerprints.add(structural_fingerprint(seed_sql))
    log_line(
        settings.log_file,
        f"path_id={path_state.path_constraint.path_id} generated_seed_index={generated_seed_index} "
        f"path_seed_index={path_state.generated_seed_count} runtime={seed_record['runtime']:.6f}s "
        f"attempts={seed_record['attempts']} elapsed={generator_elapsed:.3f}s sql={seed_sql}",
    )
    accepted_delta = expand_one_seed(
        settings=settings,
        runtime_state=runtime_state,
        seed_state=SeedState(seed_sql=seed_sql, path_constraint=path_state.path_constraint),
        seed_index=generated_seed_index,
    )
    path_state.accepted_sql_count += accepted_delta
    return True, accepted_delta, session.agent


def should_exhaust_path(settings: ExpanderSettings, path_state: PathRuntimeState) -> tuple[bool, str]:

    if path_state.generated_seed_count == 0 and path_state.failed_seed_count > 0:
        return True, "First round produced no seed"
    if path_state.generated_seed_count >= settings.live_max_seeds_per_path:
        return True, "Path seed budget reached"
    if path_state.no_gain_turns >= settings.live_max_no_gain_turns:
        return True, "No gain across consecutive rounds"
    return False, ""


def run_live_workflow(*, emit_seed_output: bool = False) -> dict:

    settings = load_settings()
    checkpoint = load_checkpoint(settings)
    if (
        checkpoint is not None
        and checkpoint.get("status") == "finished"
        and int(checkpoint.get("accepted_count", 0)) < settings.target_count
    ):
        checkpoint = None
    fresh_start = not settings.resume_run
    if checkpoint is None:
        files_to_reset = [settings.log_file]
        if fresh_start:
            files_to_reset = [
                settings.seed_file,
                settings.output_file,
                settings.records_file,
                settings.log_file,
                settings.timing_log_file,
                settings.checkpoint_file,
                settings.planner_paths_file,
                settings.planner_constraints_file,
            ]
        reset_run_files(
            *files_to_reset,
        )
        if settings.resume_run:
            log_line(
                settings.log_file,
                "live_workflow_resume_without_checkpoint "
                "records will be restored and planner will be rebuilt",
            )
        log_line(
            settings.log_file,
            f"live_workflow_prepare target_count={settings.target_count} planner_max_paths={settings.planner_max_paths}",
        )
        log_line(settings.log_file, "planner_start")

        plan_bundle = build_plan_bundle(
            summary_file=settings.summary_file,
            sample_value_limit=settings.sample_value_limit,
            max_paths=settings.planner_max_paths,
            runtime_window=settings.runtime_window,
        )
        save_plan_outputs(
            list(plan_bundle.paths),
            settings.planner_paths_file,
            settings.planner_constraints_file,
        )
        log_line(
            settings.log_file,
            "planner_finished "
            f"candidate_count={plan_bundle.candidate_count} "
            f"bucket_counts={plan_bundle.candidate_bucket_counts} "
            f"selected_path_count={len(plan_bundle.paths)}",
        )
        for path_index, constraint in enumerate(plan_bundle.paths, start=1):
            log_line(
                settings.log_file,
                "planner_selected "
                f"path_index={path_index} path_id={constraint.path_id} "
                f"path_name={constraint.path_name} from_mode={constraint.from_mode} "
                f"allowed_tables={list(constraint.allowed_tables)} "
                f"allowed_join_paths={list(constraint.allowed_join_paths)}",
            )
        diversity_source_count = len(plan_bundle.paths)
    else:
        log_line(
            settings.log_file,
            f"live_workflow_resume target_count={settings.target_count} "
            f"checkpoint={settings.checkpoint_file} "
            f"checkpoint_accepted_count={checkpoint.get('accepted_count', 0)}",
        )
        plan_bundle = None
        diversity_source_count = len(checkpoint.get("paths", [])) or settings.planner_max_paths
    log_line(settings.log_file, "runtime_state_start")

    runtime_state: ExpanderRuntimeState = build_runtime_state(
        settings,
        [],
        reset_outputs=False,
        diversity_source_count=diversity_source_count,
    )
    if checkpoint is not None:
        plan_bundle = load_plan_bundle_from_checkpoint(settings, runtime_state.summary, checkpoint)
        log_line(
            settings.log_file,
            f"runtime_state_restored accepted_count={len(runtime_state.accepted_results)} "
            f"planner_path_count={len(plan_bundle.paths)}",
        )
        if len(runtime_state.accepted_results) >= settings.target_count:
            close_runtime_state(runtime_state)
            return {
                "generated_seed_count": int(checkpoint.get("generated_seed_count", 0)),
                "accepted_count": len(runtime_state.accepted_results),
                "seed_file": str(settings.seed_file),
                "output_file": str(settings.output_file),
                "records_file": str(settings.records_file),
                "log_file": str(settings.log_file),
                "timing_log_file": str(settings.timing_log_file),
                "planner_paths_file": str(settings.planner_paths_file),
                "planner_constraints_file": str(settings.planner_constraints_file),
                "used_path_count": 0,
                "path_run_summaries": [],
            }
    log_line(settings.log_file, "runtime_state_finished")
    generated_seed_count = int(checkpoint.get("generated_seed_count", 0)) if checkpoint else 0
    global_turn_index = int(checkpoint.get("global_turn_index", 0)) if checkpoint else 0
    round_index = int(checkpoint.get("round_index", 0)) if checkpoint else 0
    shared_agent = None

    if checkpoint is None:
        path_states = [
            PathRuntimeState(path_constraint=path_constraint, accept_limit=runtime_state.path_accept_limit)
            for path_constraint in plan_bundle.paths
        ]
        ready_queue = deque(range(len(path_states)))
    else:
        path_states = restore_path_states(checkpoint, plan_bundle, runtime_state.path_accept_limit)
        ready_queue = deque(
            index
            for index in checkpoint.get("ready_queue", [])
            if isinstance(index, int) and 0 <= index < len(path_states) and path_states[index].status != "exhausted"
        )
        if not ready_queue and len(runtime_state.accepted_results) < settings.target_count:
            ready_queue = deque(
                index for index, path_state in enumerate(path_states) if path_state.status != "exhausted"
            )
    for path_state in path_states:
        restored_path_count = runtime_state.accepted_path_counts.get(path_state.path_constraint.path_id, 0)
        path_state.accepted_sql_count = max(path_state.accepted_sql_count, restored_path_count)

    try:
        log_line(
            settings.log_file,
                f"live_workflow_start target_count={settings.target_count} planner_path_count={len(plan_bundle.paths)}",
            )
        for path_index, path_state in enumerate(path_states, start=1):
            constraint = path_state.path_constraint
            log_line(
                settings.log_file,
                "path_registered "
                f"path_index={path_index} path_id={constraint.path_id} "
                f"path_name={constraint.path_name} from_mode={constraint.from_mode} "
                f"allowed_tables={list(constraint.allowed_tables)} "
                f"accept_limit={path_state.accept_limit}",
            )
        save_checkpoint(
            settings,
            plan_bundle,
            path_states,
            ready_queue,
            generated_seed_count=generated_seed_count,
            global_turn_index=global_turn_index,
            round_index=round_index,
            accepted_count=len(runtime_state.accepted_results),
        )



        while ready_queue and len(runtime_state.accepted_results) < settings.target_count:
            round_index += 1
            current_round = list(ready_queue)
            ready_queue.clear()
            round_accepted_before = len(runtime_state.accepted_results)
            log_line(
                settings.log_file,
                f"round_start round={round_index} path_count={len(current_round)} "
                f"accepted_count={len(runtime_state.accepted_results)}",
            )

            for round_path_offset, path_index in enumerate(current_round):
                if len(runtime_state.accepted_results) >= settings.target_count:
                    break

                path_state = path_states[path_index]
                constraint = path_state.path_constraint

                if path_state.status == "exhausted":
                    continue

                save_checkpoint(
                    settings,
                    plan_bundle,
                    path_states,
                    list(current_round[round_path_offset:]) + list(ready_queue),
                    generated_seed_count=generated_seed_count,
                    global_turn_index=global_turn_index,
                    round_index=round_index,
                    accepted_count=len(runtime_state.accepted_results),
                )

                global_turn_index += 1
                path_state.turn_count += 1
                path_state.last_selected_turn = global_turn_index
                path_state.status = "running"
                accepted_before_turn = len(runtime_state.accepted_results)
                seeds_generated_this_turn = 0

                log_line(
                    settings.log_file,
                    f"path_turn_start round={round_index} turn={global_turn_index} path_id={constraint.path_id} "
                    f"path_name={constraint.path_name} generated_seed_count={path_state.generated_seed_count} "
                    f"accepted_sql_count={path_state.accepted_sql_count} no_gain_turns={path_state.no_gain_turns}",
                )

                while (
                    len(runtime_state.accepted_results) < settings.target_count
                    and seeds_generated_this_turn < settings.live_seeds_per_turn
                    and path_state.generated_seed_count < settings.live_max_seeds_per_path
                ):
                    accepted_delta = 0
                    produced_seed = False

                    try:
                        produced_seed, accepted_delta, shared_agent = run_generator_turn(
                            settings=settings,
                            runtime_state=runtime_state,
                            path_state=path_state,
                            generated_seed_index=generated_seed_count + 1,
                            summary=plan_bundle.summary,
                            shared_agent=shared_agent,
                            emit_output=emit_seed_output,
                        )
                    except Exception as exc:
                        path_state.failed_seed_count += 1
                        log_line(
                            settings.log_file,
                            f"path_id={constraint.path_id} path_turn_error "
                            f"round={round_index} turn={global_turn_index} "
                            f"error_type={type(exc).__name__} error={exc}",
                        )
                        for trace_line in traceback.format_exc().splitlines():
                            log_line(
                                settings.log_file,
                                f"path_id={constraint.path_id} path_turn_trace {trace_line}",
                            )
                        break

                    if not produced_seed:
                        break

                    generated_seed_count += 1
                    seeds_generated_this_turn += 1
                    if accepted_delta > 0:
                        path_state.last_gain_turn = global_turn_index

                accepted_gain = len(runtime_state.accepted_results) - accepted_before_turn
                if accepted_gain > 0:
                    path_state.no_gain_turns = 0
                else:
                    path_state.no_gain_turns += 1

                exhaust, reason = should_exhaust_path(settings, path_state)
                if exhaust:
                    path_state.status = "exhausted"
                    log_line(
                        settings.log_file,
                        f"path_exhausted round={round_index} turn={global_turn_index} "
                        f"path_id={constraint.path_id} reason={reason} "
                        f"generated_seed_count={path_state.generated_seed_count} "
                        f"accepted_sql_count={path_state.accepted_sql_count}",
                    )
                else:
                    path_state.status = "ready"
                    ready_queue.append(path_index)
                    log_line(
                        settings.log_file,
                        f"path_requeue round={round_index} turn={global_turn_index} "
                        f"path_id={constraint.path_id} generated_seed_count={path_state.generated_seed_count} "
                        f"accepted_sql_count={path_state.accepted_sql_count} "
                        f"no_gain_turns={path_state.no_gain_turns}",
                    )
                save_checkpoint(
                    settings,
                    plan_bundle,
                    path_states,
                    list(current_round[round_path_offset + 1 :]) + list(ready_queue),
                    generated_seed_count=generated_seed_count,
                    global_turn_index=global_turn_index,
                    round_index=round_index,
                    accepted_count=len(runtime_state.accepted_results),
                )

            round_accepted_gain = len(runtime_state.accepted_results) - round_accepted_before
            log_line(
                settings.log_file,
                f"round_end round={round_index} accepted_gain={round_accepted_gain} "
                f"accepted_count={len(runtime_state.accepted_results)} remaining_paths={len(ready_queue)}",
            )
            save_checkpoint(
                settings,
                plan_bundle,
                path_states,
                ready_queue,
                generated_seed_count=generated_seed_count,
                global_turn_index=global_turn_index,
                round_index=round_index,
                accepted_count=len(runtime_state.accepted_results),
            )

        path_run_summaries = []
        for path_state in path_states:
            path_run_summaries.append(
                {
                    "path_id": path_state.path_constraint.path_id,
                    "path_name": path_state.path_constraint.path_name,
                    "status": path_state.status,
                    "generated_seed_count": path_state.generated_seed_count,
                    "accepted_sql_count": path_state.accepted_sql_count,
                    "failed_seed_count": path_state.failed_seed_count,
                    "no_gain_turns": path_state.no_gain_turns,
                    "turn_count": path_state.turn_count,
                }
            )

        final_summary = {
            "generated_seed_count": generated_seed_count,
            "accepted_count": len(runtime_state.accepted_results),
            "used_path_count": len([state for state in path_states if state.generated_seed_count > 0]),
            "planner_paths_file": str(settings.planner_paths_file),
            "planner_constraints_file": str(settings.planner_constraints_file),
            "path_stats": summarize_path_stats(runtime_state),
            "path_run_summaries": path_run_summaries,
        }
        log_line(settings.log_file, "live_workflow_finished " + json.dumps(final_summary, ensure_ascii=False))
        save_checkpoint(
            settings,
            plan_bundle,
            path_states,
            ready_queue,
            generated_seed_count=generated_seed_count,
            global_turn_index=global_turn_index,
            round_index=round_index,
            accepted_count=len(runtime_state.accepted_results),
            status="finished",
        )
        return {
            "generated_seed_count": generated_seed_count,
            "accepted_count": len(runtime_state.accepted_results),
            "seed_file": str(settings.seed_file),
            "output_file": str(settings.output_file),
            "records_file": str(settings.records_file),
            "log_file": str(settings.log_file),
            "timing_log_file": str(settings.timing_log_file),
            "planner_paths_file": str(settings.planner_paths_file),
            "planner_constraints_file": str(settings.planner_constraints_file),
            "used_path_count": len([state for state in path_states if state.generated_seed_count > 0]),
            "path_run_summaries": path_run_summaries,
        }
    finally:
        close_runtime_state(runtime_state)
