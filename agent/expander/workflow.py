from dataclasses import dataclass, field
import json
import math
import re
import time
from typing import Any

from agent.planner.models import (
    PathConstraint,
    extract_sql_tables,
    sql_contains_clause,
    sql_matches_constraint,
    sql_uses_aggregate,
)
from pool import build_related_summary, connect_postgresql, load_or_build_summary

from agent.expander.config import ExpanderSettings
from agent.expander.io_utils import append_jsonl, append_sql, log_line, reset_run_files
from agent.expander.llm import call_deepseek
from agent.expander.prompting import TARGET_CAUSES, build_multi_direction_user_prompt, build_system_prompt
from agent.expander.validation import StableValidationResult, explain_plan_json, structural_fingerprint, validate_candidate_stable


@dataclass
class SeedState:

    seed_sql: str
    path_constraint: PathConstraint | None = None
    refine_depth: int = 0
    rounds: int = 0
    recent_failures: list[str] = field(default_factory=list)
    accepted_fingerprints: list[str] = field(default_factory=list)
    accepted_fast_sql_count: int = 0


@dataclass
class ExpanderRuntimeState:

    summary: dict
    connection: Any
    system_prompt: str
    attempted_fingerprints: set[str] = field(default_factory=set)
    accepted_results: list[Any] = field(default_factory=list)
    path_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    accepted_path_counts: dict[str, int] = field(default_factory=dict)
    accepted_combo_counts: dict[str, int] = field(default_factory=dict)
    accepted_structure_counts: dict[str, int] = field(default_factory=dict)
    path_accept_limit: int = 0
    combo_accept_limit: int = 0
    structure_accept_limit: int = 0
    run_started_at: float = field(default_factory=time.perf_counter)


VALID_TARGET_CAUSES = {name for name, _ in TARGET_CAUSES}
COLUMN_REF_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b")
SELECT_FROM_PATTERN = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
WHERE_BOUNDARY_PATTERN = re.compile(
    r"\b(where|having)\b(.*?)(\bgroup\s+by\b|\border\s+by\b|;|$)",
    re.IGNORECASE | re.DOTALL,
)
GROUP_ORDER_PATTERN = re.compile(
    r"\b(group\s+by|order\s+by)\b(.*?)(\bhaving\b|\border\s+by\b|;|$)",
    re.IGNORECASE | re.DOTALL,
)


def load_seed_sqls(seed_file) -> list[str]:

    seed_sqls = []
    for raw_line in seed_file.read_text(encoding="utf-8").splitlines():
        sql = raw_line.strip()
        if sql == "":
            continue
        seed_sqls.append(sql)
    deduplicated = []
    seen = set()
    for sql in seed_sqls:
        fingerprint = structural_fingerprint(sql)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduplicated.append(sql)
    return deduplicated


def build_failure_note(result) -> str:

    return (
        f"status={result.status} runtime={result.runtime_seconds:.6f}s "
        f"message={result.message} sql={result.sql}"
    )


def build_table_combo_key(sql: str, summary: dict) -> str:

    tables = extract_sql_tables(sql, summary)
    return ",".join(tables) if tables else "__unknown__"


def build_structure_signature(sql: str, summary: dict) -> str:

    normalized = structural_fingerprint(sql)
    tables = extract_sql_tables(sql, summary)
    return "|".join(
        [
            f"tables={len(tables)}",
            f"join={int(' join ' in normalized)}",
            f"agg={int(sql_uses_aggregate(normalized))}",
            f"group={int(sql_contains_clause(normalized, 'group_by'))}",
            f"having={int(sql_contains_clause(normalized, 'having'))}",
            f"order={int(sql_contains_clause(normalized, 'order_by'))}",
            f"predicate={int(sql_contains_clause(normalized, 'where') or sql_contains_clause(normalized, 'having'))}",
        ]
    )


def build_structure_summary(sql: str, summary: dict) -> dict:

    tables = extract_sql_tables(sql, summary)
    return {
        "table_count": len(tables),
        "has_join": " join " in structural_fingerprint(sql),
        "has_aggregate": sql_uses_aggregate(sql),
        "has_group_by": sql_contains_clause(sql, "group_by"),
        "has_having": sql_contains_clause(sql, "having"),
        "has_order_by": sql_contains_clause(sql, "order_by"),
    }


def extract_columns_from_text(text: str) -> set[str]:

    return {match.group(0).lower() for match in COLUMN_REF_PATTERN.finditer(text)}


def extract_projection_columns(sql: str) -> set[str]:

    match = SELECT_FROM_PATTERN.search(sql)
    if match is None:
        return set()
    return extract_columns_from_text(match.group(1))


def extract_predicate_columns(sql: str) -> set[str]:

    columns: set[str] = set()
    for match in WHERE_BOUNDARY_PATTERN.finditer(sql):
        columns |= extract_columns_from_text(match.group(2))
    return columns


def extract_group_order_columns(sql: str) -> set[str]:

    columns: set[str] = set()
    for match in GROUP_ORDER_PATTERN.finditer(sql):
        columns |= extract_columns_from_text(match.group(2))
    return columns


def build_shape_profile(sql: str, summary: dict) -> dict[str, set[str]]:

    return {
        "tables": set(extract_sql_tables(sql, summary)),
        "projection_columns": extract_projection_columns(sql),
        "predicate_columns": extract_predicate_columns(sql),
        "group_order_columns": extract_group_order_columns(sql),
    }


def jaccard_similarity(left: set[str], right: set[str]) -> float:

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def same_nonempty_set(left: set[str], right: set[str]) -> bool:

    return bool(left) and left == right


def shape_reject_reason(sql: str, runtime_state: ExpanderRuntimeState, seed_sql: str) -> str:

    candidate_shape = build_shape_profile(sql, runtime_state.summary)
    seed_shape = build_shape_profile(seed_sql, runtime_state.summary)
    if (
        same_nonempty_set(candidate_shape["projection_columns"], seed_shape["projection_columns"])
        and same_nonempty_set(candidate_shape["predicate_columns"], seed_shape["predicate_columns"])
        and same_nonempty_set(candidate_shape["group_order_columns"], seed_shape["group_order_columns"])
    ):
        return "Too similar to the seed query skeleton"

    for accepted_result in runtime_state.accepted_results:
        accepted_shape = build_shape_profile(accepted_result.sql, runtime_state.summary)
        table_similarity = jaccard_similarity(candidate_shape["tables"], accepted_shape["tables"])
        projection_similarity = jaccard_similarity(
            candidate_shape["projection_columns"],
            accepted_shape["projection_columns"],
        )
        predicate_same = same_nonempty_set(
            candidate_shape["predicate_columns"],
            accepted_shape["predicate_columns"],
        )
        group_order_same = same_nonempty_set(
            candidate_shape["group_order_columns"],
            accepted_shape["group_order_columns"],
        )
        if table_similarity >= 0.8 and projection_similarity >= 0.8 and (predicate_same or group_order_same):
            return "Too similar to an accepted SQL query skeleton"
    return ""


def iter_plan_nodes(plan_payload: object):

    if isinstance(plan_payload, list):
        for item in plan_payload:
            yield from iter_plan_nodes(item)
        return
    if not isinstance(plan_payload, dict):
        return
    if "Plan" in plan_payload:
        yield from iter_plan_nodes(plan_payload["Plan"])
        return
    if "Node Type" in plan_payload:
        yield plan_payload
    for child in plan_payload.get("Plans", []):
        yield from iter_plan_nodes(child)


def node_type_contains(plan_nodes: list[dict], *needles: str) -> bool:

    lowered_needles = tuple(needle.lower() for needle in needles)
    for node in plan_nodes:
        node_type = str(node.get("Node Type", "")).lower()
        if any(needle in node_type for needle in lowered_needles):
            return True
    return False


def join_node_count(plan_nodes: list[dict]) -> int:

    return sum(1 for node in plan_nodes if "join" in str(node.get("Node Type", "")).lower())


def has_index_scan(plan_nodes: list[dict]) -> bool:

    return node_type_contains(plan_nodes, "index scan", "index only scan", "bitmap index scan")


def has_scan_filter(plan_nodes: list[dict]) -> bool:

    for node in plan_nodes:
        node_type = str(node.get("Node Type", "")).lower()
        if "scan" in node_type and "Filter" in node:
            return True
    return False


def has_seq_scan(plan_nodes: list[dict]) -> bool:

    return node_type_contains(plan_nodes, "seq scan")


def has_subquery_structure(sql: str, plan_nodes: list[dict]) -> bool:

    normalized = structural_fingerprint(sql)
    if any(token in normalized for token in (" exists ", " in (select ", " from (select ", " = (select ")):
        return True
    return node_type_contains(plan_nodes, "subquery scan", "initplan", "subplan")


def has_inefficient_predicate(sql: str, plan_nodes: list[dict]) -> bool:

    normalized = structural_fingerprint(sql)
    predicate_tokens = (
        " like ",
        " not like ",
        " not in ",
        " <> ",
        " != ",
        " or ",
        " between ",
    )
    if any(token in normalized for token in predicate_tokens):
        return True
    return has_scan_filter(plan_nodes) and not has_index_scan(plan_nodes)


def has_high_cost_operator(plan_nodes: list[dict]) -> bool:

    return node_type_contains(plan_nodes, "sort", "aggregate") or join_node_count(plan_nodes) > 0


def score_index_cause(sql: str, plan_nodes: list[dict]) -> tuple[int, str]:

    normalized = structural_fingerprint(sql)
    if not (sql_contains_clause(normalized, "where") or sql_contains_clause(normalized, "having")):
        return 0, "Missing WHERE/HAVING predicate"
    score = 0
    evidence = []
    if has_seq_scan(plan_nodes) and has_scan_filter(plan_nodes):
        score += 5
        evidence.append("Plan contains Seq Scan with filter")
    elif has_scan_filter(plan_nodes) and not has_index_scan(plan_nodes):
        score += 4
        evidence.append("Plan filters rows without using an index scan")
    if not has_index_scan(plan_nodes):
        score += 1
        evidence.append("No index scan used")
    high_cost_penalty = join_node_count(plan_nodes)
    if node_type_contains(plan_nodes, "sort", "aggregate"):
        high_cost_penalty += 1
    score -= min(3, high_cost_penalty)
    if not evidence:
        return 0, "No inefficient scan filtering evidence"
    return max(0, score), "; ".join(evidence)


def score_predicate_cause(sql: str, plan_nodes: list[dict]) -> tuple[int, str]:

    normalized = structural_fingerprint(sql)
    score = 0
    evidence = []
    if has_subquery_structure(sql, plan_nodes):
        score += 5
        evidence.append("SQL or plan contains subquery structure")
    predicate_tokens = {
        " like ": "LIKE predicate",
        " not like ": "NOT LIKE predicate",
        " not in ": "NOT IN predicate",
        " <> ": "non-equality predicate",
        " != ": "non-equality predicate",
        " or ": "OR predicate",
        " between ": "BETWEEN predicate",
    }
    matched_predicates = [name for token, name in predicate_tokens.items() if token in normalized]
    if matched_predicates:
        score += 4
        evidence.append("Contains " + "/".join(sorted(set(matched_predicates))))
    if has_scan_filter(plan_nodes) and not has_index_scan(plan_nodes):
        score += 1
        evidence.append("Plan filters rows without an index scan")
    if not evidence:
        return 0, "No inefficient predicate or subquery evidence"
    return score, "; ".join(evidence)


def score_high_cost_cause(sql: str, plan_nodes: list[dict]) -> tuple[int, str]:

    normalized = structural_fingerprint(sql)
    score = 0
    evidence = []
    joins = join_node_count(plan_nodes)
    if joins > 0:
        score += min(5, joins * 2)
        evidence.append(f"Plan contains {joins} join operator(s)")
    if node_type_contains(plan_nodes, "sort"):
        score += 3
        evidence.append("Plan contains Sort")
    if node_type_contains(plan_nodes, "aggregate"):
        score += 3
        evidence.append("Plan contains Aggregate")
    if sql_contains_clause(normalized, "order_by"):
        score += 1
        evidence.append("SQL contains ORDER BY")
    if sql_contains_clause(normalized, "group_by") or sql_uses_aggregate(normalized):
        score += 1
        evidence.append("SQL contains GROUP BY or aggregation")
    if " join " in normalized:
        score += 1
        evidence.append("SQL contains explicit JOIN")
    if not evidence:
        return 0, "No sort / aggregation / join evidence"
    return score, "; ".join(evidence)


def relabel_target_cause(connection, sql: str, requested_cause: str) -> tuple[bool, str, str]:

    if requested_cause not in VALID_TARGET_CAUSES:
        return False, requested_cause, f"Unknown slow cause {requested_cause}"

    explain_ok, plan_payload, explain_message = explain_plan_json(connection, sql)
    if not explain_ok:
        return False, requested_cause, f"EXPLAIN JSON failed: {explain_message}"
    plan_nodes = list(iter_plan_nodes(plan_payload))
    scored_causes = [
        ("high_sort_aggregation_join_cost", *score_high_cost_cause(sql, plan_nodes)),
        ("inefficient_predicate_or_complex_subquery", *score_predicate_cause(sql, plan_nodes)),
        ("index_missing_or_inappropriate", *score_index_cause(sql, plan_nodes)),
    ]
    scored_causes.sort(key=lambda item: item[1], reverse=True)
    label, score, evidence = scored_causes[0]
    if score <= 0:
        diagnostics = "; ".join(f"{name}: {message}" for name, _, message in scored_causes)
        return False, requested_cause, f"No evidence for any slow-cause class: {diagnostics}"
    relabel_message = (
        f"tool_relabel={label} score={score} evidence={evidence} "
        f"llm_label={requested_cause}"
    )
    return True, label, relabel_message


def build_label_reason(
    label: str,
    candidate: dict,
    result,
) -> str:

    change_summary = candidate.get("change_summary", "").strip()
    reason_parts = []
    if change_summary:
        reason_parts.append(change_summary)
    reason_parts.append(f"target_cause={candidate['target_cause']}")
    reason_parts.append(f"diagnostic_label={label}")
    relabel_message = candidate.get("relabel_message", "").strip()
    if relabel_message:
        reason_parts.append(relabel_message)
    reason_parts.append(f"runtime_mean={result.runtime_seconds:.6f}s")
    reason_parts.append(
        "repeat_runtime_range="
        f"[{result.runtime_min:.6f}s, {result.runtime_max:.6f}s]"
    )
    reason_parts.append(f"runtime_band={result.runtime_band}")
    return "; ".join(reason_parts) + "."


def build_record(
    seed_state: SeedState,
    path_id: str,
    path_name: str,
    candidate: dict,
    result,
    runtime_state: ExpanderRuntimeState,
) -> dict:

    sql = result.sql
    label = candidate.get("verified_target_cause", candidate["target_cause"])
    return {
        "sql_text": sql,
        "path_id": path_id,
        "business_path": path_name,
        "runtime": result.runtime_seconds,
        "runtime_band": result.runtime_band,
        "stability_status": "stable" if result.status == "ok" else result.status,
        "repeat_execution_stats": {
            "min": result.runtime_min,
            "max": result.runtime_max,
            "mean": result.runtime_seconds,
            "std": result.runtime_std,
        },
        "diagnostic_label": label,
        "llm_original_label": candidate["target_cause"],
        "label_source": "generation_tool_relabel",
        "tool_label_evidence": candidate.get("relabel_message", ""),
        "label_reason": build_label_reason(
            label=label,
            candidate=candidate,
            result=result,
        ),
        "seed_sql": seed_state.seed_sql,
        "source_type": "expand",
        "structure_summary": build_structure_summary(sql, runtime_state.summary),
        "query_skeleton": {
            "query_shape": candidate.get("query_shape", ""),
            "main_projection_columns": candidate.get("main_projection_columns", []),
            "main_predicate_columns": candidate.get("main_predicate_columns", []),
            "main_group_order_columns": candidate.get("main_group_order_columns", []),
        },
        "generation_round": seed_state.rounds,
    }


def compute_accept_limits(settings: ExpanderSettings, diversity_source_count: int) -> tuple[int, int, int]:

    source_count = max(1, diversity_source_count)
    path_limit = 0
    combo_limit = max(8, math.ceil(settings.target_count / source_count * 3.0))
    structure_limit = max(8, math.ceil(settings.target_count / source_count * 2.4))
    return path_limit, combo_limit, structure_limit


def relaxed_limit(base_limit: int, accepted_count: int, target_count: int) -> int:

    if target_count <= 0:
        return base_limit
    progress = accepted_count / target_count
    if progress >= 0.9:
        return base_limit + max(8, base_limit)
    if progress >= 0.75:
        return base_limit + max(4, base_limit // 2)
    return base_limit


def diversity_reject_reason(
    settings: ExpanderSettings,
    runtime_state: ExpanderRuntimeState,
    path_id: str,
    sql: str,
) -> str:

    accepted_count = len(runtime_state.accepted_results)
    combo_limit = relaxed_limit(runtime_state.combo_accept_limit, accepted_count, settings.target_count)
    structure_limit = relaxed_limit(runtime_state.structure_accept_limit, accepted_count, settings.target_count)

    combo_key = build_table_combo_key(sql, runtime_state.summary)
    if runtime_state.accepted_combo_counts.get(combo_key, 0) >= combo_limit:
        return "Table-combination quota is full"

    structure_key = build_structure_signature(sql, runtime_state.summary)
    if runtime_state.accepted_structure_counts.get(structure_key, 0) >= structure_limit:
        return "Structure quota is full"
    return ""


def register_accepted_sql(runtime_state: ExpanderRuntimeState, path_id: str, sql: str) -> None:

    combo_key = build_table_combo_key(sql, runtime_state.summary)
    structure_key = build_structure_signature(sql, runtime_state.summary)
    runtime_state.accepted_path_counts[path_id] = runtime_state.accepted_path_counts.get(path_id, 0) + 1
    runtime_state.accepted_combo_counts[combo_key] = runtime_state.accepted_combo_counts.get(combo_key, 0) + 1
    runtime_state.accepted_structure_counts[structure_key] = (
        runtime_state.accepted_structure_counts.get(structure_key, 0) + 1
    )


def write_workload_record(
    settings: ExpanderSettings,
    runtime_state: ExpanderRuntimeState,
    seed_state: SeedState,
    path_id: str,
    path_name: str,
    candidate: dict,
    result,
) -> None:

    append_sql(settings.output_file, result.sql)
    accepted_index = len(runtime_state.accepted_results) + 1
    append_jsonl(
        settings.records_file,
        build_record(
            seed_state=seed_state,
            path_id=path_id,
            path_name=path_name,
            candidate=candidate,
            result=result,
            runtime_state=runtime_state,
        ),
    )
    append_jsonl(
        settings.timing_log_file,
        {
            "index": accepted_index,
            "time_seconds": time.perf_counter() - runtime_state.run_started_at,
            "sql": result.sql,
        },
    )
    register_accepted_sql(runtime_state, path_id, result.sql)
    seed_state.accepted_fingerprints.append(result.fingerprint)
    if result.runtime_band == "fast":
        seed_state.accepted_fast_sql_count += 1
    runtime_state.accepted_results.append(result)


def restore_accepted_records(settings: ExpanderSettings, runtime_state: ExpanderRuntimeState) -> int:

    if not settings.records_file.exists():
        return 0

    restored_count = 0
    for raw_line in settings.records_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        sql = record.get("sql_text", record.get("SQL\u6587\u672c"))
        path_id = record.get("path_id", record.get("\u8def\u5f84ID", "default_path"))
        runtime = float(record.get("runtime", 0.0))
        stats = record.get("repeat_execution_stats", record.get("\u91cd\u590d\u6267\u884c\u7edf\u8ba1", {}))
        result = StableValidationResult(
            sql=sql,
            fingerprint=structural_fingerprint(sql),
            status=record.get("stability_status", record.get("\u7a33\u5b9a\u6027\u72b6\u6001", "restored")),
            runtime_seconds=runtime,
            runtime_min=float(stats.get("min", stats.get("\u6700\u5c0f\u503c", runtime))),
            runtime_max=float(stats.get("max", stats.get("\u6700\u5927\u503c", runtime))),
            runtime_std=float(stats.get("std", stats.get("\u6807\u51c6\u5dee", 0.0))),
            runtime_band=record.get("runtime_band", record.get("runtime\u5206\u5c42", "")),
            accepted=True,
            message="restored_from_records",
            run_count=0,
        )
        runtime_state.attempted_fingerprints.add(result.fingerprint)
        register_accepted_sql(runtime_state, path_id, sql)
        runtime_state.accepted_results.append(result)
        restored_count += 1
    return restored_count


def path_limit_reached(settings: ExpanderSettings, runtime_state: ExpanderRuntimeState, path_id: str) -> bool:

    return False


def classify_reject_reason(result, runtime_window) -> str:

    if result.status == "timeout":
        return "Execution timed out"
    if result.status == "over_budget":
        return "Exceeded early-stop budget"
    if result.status == "unstable":
        return "Runtime is unstable"
    if result.status == "explain_error":
        return "EXPLAIN failed"
    if result.status == "error":
        return "Execution error"
    if result.status == "non_select":
        return "Not a SELECT statement"
    return "No hit"


def request_candidates_for_seed_round(
    settings: ExpanderSettings,
    runtime_state: ExpanderRuntimeState,
    seed_state: SeedState,
    related_summary: dict,
) -> list[dict]:

    total_candidate_limit = len(TARGET_CAUSES)
    user_prompt = build_multi_direction_user_prompt(
        seed_sql=seed_state.seed_sql,
        related_summary=related_summary,
        runtime_window=settings.runtime_window,
        total_candidate_limit=total_candidate_limit,
        recent_failures=seed_state.recent_failures[-3:],
        recent_success_fingerprints=seed_state.accepted_fingerprints[-3:],
        path_context=seed_state.path_constraint.to_prompt_dict() if seed_state.path_constraint else None,
        refine_depth=seed_state.refine_depth,
    )
    payload, _ = call_deepseek(settings, runtime_state.system_prompt, user_prompt)
    return payload["candidates"][:total_candidate_limit]


def final_accept_reject_reason(result, runtime_window) -> str:


    if result.runtime_band == "fast":
        return "Fast candidates are only used as intermediate states for later slow-down refinement"
    if result.runtime_seconds > runtime_window.slow_accept_max_seconds:
        return "Exceeded the acceptable slow upper bound"
    return ""


def build_runtime_state(
    settings: ExpanderSettings,
    initial_seed_sqls: list[str],
    *,
    reset_outputs: bool,
    diversity_source_count: int | None = None,
) -> ExpanderRuntimeState:

    if reset_outputs:
        reset_run_files(settings.output_file, settings.records_file, settings.log_file, settings.timing_log_file)
    summary = load_or_build_summary(settings.summary_file, settings.sample_value_limit)
    attempted_fingerprints = {structural_fingerprint(seed_sql) for seed_sql in initial_seed_sqls}
    path_limit, combo_limit, structure_limit = compute_accept_limits(
        settings,
        diversity_source_count or len(initial_seed_sqls) or settings.planner_max_paths,
    )
    runtime_state = ExpanderRuntimeState(
        summary=summary,
        connection=connect_postgresql(),
        system_prompt=build_system_prompt(),
        attempted_fingerprints=attempted_fingerprints,
        path_accept_limit=path_limit,
        combo_accept_limit=combo_limit,
        structure_accept_limit=structure_limit,
    )
    if not reset_outputs:
        restore_accepted_records(settings, runtime_state)
    return runtime_state


def close_runtime_state(runtime_state: ExpanderRuntimeState) -> None:

    runtime_state.connection.close()


def expand_one_seed(
    settings: ExpanderSettings,
    runtime_state: ExpanderRuntimeState,
    seed_state: SeedState,
    seed_index: int,
) -> int:

    accepted_before = len(runtime_state.accepted_results)
    refine_queue: list[SeedState] = []
    path_id = seed_state.path_constraint.path_id if seed_state.path_constraint else "default_path"
    path_name = seed_state.path_constraint.path_name if seed_state.path_constraint else "default_path"
    runtime_state.path_stats.setdefault(
        path_id,
        {
            "accepted": 0,
            "duplicate": 0,
            "quota_reject": 0,
            "timeout": 0,
            "fast_band": 0,
            "near_target_band": 0,
            "slow_band": 0,
            "unstable": 0,
            "explain_error": 0,
            "exec_error": 0,
            "cause_reject": 0,
            "shape_reject": 0,
            "validated": 0,
            "seed_count": 0,
        },
    )
    runtime_state.path_stats[path_id]["seed_count"] += 1
    while (
        len(runtime_state.accepted_results) < settings.target_count
        and seed_state.rounds < settings.max_rounds_per_seed
    ):
        if path_limit_reached(settings, runtime_state, path_id):
            break

        seed_state.rounds += 1
        related_summary = build_related_summary(seed_state.seed_sql, runtime_state.summary)
        round_stats = {
            "requested": 1,
            "returned": 0,
            "validated": 0,
            "accepted": 0,
            "duplicate": 0,
            "quota_reject": 0,
            "timeout": 0,
            "fast_band": 0,
            "near_target_band": 0,
            "slow_band": 0,
            "unstable": 0,
            "explain_error": 0,
            "exec_error": 0,
            "cause_reject": 0,
            "shape_reject": 0,
        }
        target_cause_names = ",".join(cause_name for cause_name, _ in TARGET_CAUSES)
        log_line(
            settings.log_file,
            f"path_id={path_id} path_name={path_name} seed_index={seed_index} "
            f"round={seed_state.rounds} target_causes={target_cause_names} request_start",
        )

        round_candidates = request_candidates_for_seed_round(
            settings=settings,
            runtime_state=runtime_state,
            seed_state=seed_state,
            related_summary=related_summary,
        )

        if not round_candidates:
            seed_state.recent_failures.append("llm_empty_candidates")
            log_line(
                settings.log_file,
                f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} empty_candidates",
            )
            continue
        round_stats["returned"] = len(round_candidates)


        for candidate_index, candidate in enumerate(round_candidates, start=1):
            target_cause = candidate["target_cause"]
            sql = candidate["sql"]
            fingerprint = structural_fingerprint(sql)
            if fingerprint in runtime_state.attempted_fingerprints:
                round_stats["duplicate"] += 1
                runtime_state.path_stats[path_id]["duplicate"] += 1
                log_line(
                    settings.log_file,
                    f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                    f"candidate={candidate_index} status=duplicate accepted=False reason=structural_duplicate",
                )
                continue

            if seed_state.path_constraint is not None:
                matches_constraint, constraint_message = sql_matches_constraint(
                    sql=sql,
                    summary=runtime_state.summary,
                    constraint=seed_state.path_constraint,
                )
                if not matches_constraint:
                    round_stats["exec_error"] += 1
                    runtime_state.path_stats[path_id]["exec_error"] += 1
                    log_line(
                        settings.log_file,
                        f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                        f"candidate={candidate_index} status=constraint_reject accepted=False reason={constraint_message}",
                    )
                    seed_state.recent_failures.append(
                        f"status=constraint_reject message={constraint_message} sql={sql}"
                    )
                    continue

            shape_reason = shape_reject_reason(sql, runtime_state, seed_state.seed_sql)
            if shape_reason:
                round_stats["shape_reject"] += 1
                runtime_state.path_stats[path_id]["shape_reject"] += 1
                log_line(
                    settings.log_file,
                    f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                    f"candidate={candidate_index} status=shape_reject accepted=False reason={shape_reason}",
                )
                seed_state.recent_failures.append(
                    f"status=shape_reject target_cause={target_cause} message={shape_reason} sql={sql}"
                )
                continue

            runtime_state.attempted_fingerprints.add(fingerprint)

            cause_ok, verified_target_cause, cause_message = relabel_target_cause(
                runtime_state.connection,
                sql,
                target_cause,
            )
            if not cause_ok:
                round_stats["cause_reject"] += 1
                runtime_state.path_stats[path_id]["cause_reject"] += 1
                log_line(
                    settings.log_file,
                    f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                    f"candidate={candidate_index} status=cause_reject accepted=False reason={cause_message}",
                )
                seed_state.recent_failures.append(
                    f"status=cause_reject target_cause={target_cause} message={cause_message} sql={sql}"
                )
                continue
            candidate["verified_target_cause"] = verified_target_cause
            candidate["relabel_message"] = cause_message

            reject_reason = diversity_reject_reason(settings, runtime_state, path_id, sql)
            if reject_reason:
                round_stats["quota_reject"] += 1
                runtime_state.path_stats[path_id]["quota_reject"] += 1
                log_line(
                    settings.log_file,
                    f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                    f"candidate={candidate_index} status=quota_reject accepted=False reason={reject_reason}",
                )
                seed_state.recent_failures.append(f"status=quota_reject message={reject_reason} sql={sql}")
                continue

            result = validate_candidate_stable(
                connection=runtime_state.connection,
                sql=sql,
                runtime_window=settings.runtime_window,
                run_count=settings.validate_runs,
                max_runtime_cv=settings.max_runtime_cv,
                max_runtime_spread=settings.max_runtime_spread,
            )
            round_stats["validated"] += 1
            runtime_state.path_stats[path_id]["validated"] += 1
            reject_reason = ""
            if result.accepted:
                round_stats["accepted"] += 1
                runtime_state.path_stats[path_id]["accepted"] += 1
                if result.runtime_band == "fast":
                    round_stats["fast_band"] += 1
                    runtime_state.path_stats[path_id]["fast_band"] += 1
                elif result.runtime_band == "near_target":
                    round_stats["near_target_band"] += 1
                    runtime_state.path_stats[path_id]["near_target_band"] += 1
                elif result.runtime_band == "slow":
                    round_stats["slow_band"] += 1
                    runtime_state.path_stats[path_id]["slow_band"] += 1
            elif result.status == "timeout":
                round_stats["timeout"] += 1
                runtime_state.path_stats[path_id]["timeout"] += 1
                reject_reason = "Execution timed out; rejected from final workload"
            elif result.status == "over_budget":
                round_stats["timeout"] += 1
                runtime_state.path_stats[path_id]["timeout"] += 1
                reject_reason = "Exceeded early-stop budget; rejected from final workload"
            elif result.status == "unstable":
                round_stats["unstable"] += 1
                runtime_state.path_stats[path_id]["unstable"] += 1
                reject_reason = classify_reject_reason(result, settings.runtime_window)
            elif result.status == "explain_error":
                round_stats["explain_error"] += 1
                runtime_state.path_stats[path_id]["explain_error"] += 1
                reject_reason = "EXPLAIN failed"
            elif result.status in {"error", "non_select"}:
                round_stats["exec_error"] += 1
                runtime_state.path_stats[path_id]["exec_error"] += 1
                reject_reason = classify_reject_reason(result, settings.runtime_window)
            else:
                reject_reason = classify_reject_reason(result, settings.runtime_window)
            log_line(
                settings.log_file,
                f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                f"candidate={candidate_index} status={result.status} runtime={result.runtime_seconds:.6f}s "
                f"runtime_band={result.runtime_band} verified_target_cause={candidate.get('verified_target_cause', target_cause)} "
                f"accepted={result.accepted}"
                + (f" reason={reject_reason}" if not result.accepted else ""),
            )
            if result.accepted:
                final_reject_reason = final_accept_reject_reason(result, settings.runtime_window)
                if final_reject_reason:
                    round_stats["accepted"] -= 1
                    runtime_state.path_stats[path_id]["accepted"] -= 1
                    if result.runtime_band == "fast":
                        round_stats["fast_band"] -= 1
                        runtime_state.path_stats[path_id]["fast_band"] -= 1
                    elif result.runtime_band == "near_target":
                        round_stats["near_target_band"] -= 1
                        runtime_state.path_stats[path_id]["near_target_band"] -= 1
                    elif result.runtime_band == "slow":
                        round_stats["slow_band"] -= 1
                        runtime_state.path_stats[path_id]["slow_band"] -= 1
                    if result.runtime_band == "fast" and seed_state.accepted_fast_sql_count < 1:
                        seed_state.accepted_fast_sql_count += 1
                        seed_state.accepted_fingerprints.append(result.fingerprint)
                        if seed_state.refine_depth < 1:
                            refine_queue.append(
                                SeedState(
                                    seed_sql=result.sql,
                                    path_constraint=seed_state.path_constraint,
                                    refine_depth=seed_state.refine_depth + 1,
                                )
                            )
                            log_line(
                                settings.log_file,
                                f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} "
                                f"target_cause={target_cause} candidate={candidate_index} "
                                f"status=fast_refine_enqueued runtime={result.runtime_seconds:.6f}s sql={result.sql}",
                            )
                        else:
                            log_line(
                                settings.log_file,
                                f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} "
                                f"target_cause={target_cause} candidate={candidate_index} "
                                f"status=fast_refine_skipped accepted=False reason=refine_depth_limit_reached",
                            )
                    else:
                        round_stats["quota_reject"] += 1
                        runtime_state.path_stats[path_id]["quota_reject"] += 1
                        log_line(
                            settings.log_file,
                            f"path_id={path_id} seed_index={seed_index} round={seed_state.rounds} target_cause={target_cause} "
                            f"candidate={candidate_index} status=final_reject accepted=False reason={final_reject_reason}",
                        )
                        seed_state.recent_failures.append(
                            f"status=final_reject message={final_reject_reason} sql={result.sql}"
                        )
                    continue

                write_workload_record(
                    settings=settings,
                    runtime_state=runtime_state,
                    seed_state=seed_state,
                    path_id=path_id,
                    path_name=path_name,
                    candidate=candidate,
                    result=result,
                )
                log_line(
                    settings.log_file,
                    f"path_id={path_id} seed_index={seed_index} "
                    f"accepted_count={len(runtime_state.accepted_results)} runtime_band={result.runtime_band} "
                    f"llm_target_cause={target_cause} verified_target_cause={candidate.get('verified_target_cause', target_cause)} "
                    f"sql={result.sql}",
                )
                if path_limit_reached(settings, runtime_state, path_id):
                    break
                if len(runtime_state.accepted_results) >= settings.target_count:
                    break
            elif result.status in {"timeout", "over_budget"}:
                seed_state.recent_failures.append(build_failure_note(result))
            else:
                seed_state.recent_failures.append(build_failure_note(result))
            if len(runtime_state.accepted_results) >= settings.target_count:
                break
            if path_limit_reached(settings, runtime_state, path_id):
                break
        log_line(
            settings.log_file,
            f"path_id={path_id} path_name={path_name} seed_index={seed_index} round={seed_state.rounds} summary "
            f"request_count={round_stats['requested']} slow_cause_count={len(TARGET_CAUSES)} returned_candidates={round_stats['returned']} "
            f"validated={round_stats['validated']} accepted={round_stats['accepted']} "
            f"duplicate={round_stats['duplicate']} quota_reject={round_stats['quota_reject']} timeout={round_stats['timeout']} "
            f"fast={round_stats['fast_band']} near_target={round_stats['near_target_band']} slow={round_stats['slow_band']} "
            f"unstable={round_stats['unstable']} "
            f"cause_reject={round_stats['cause_reject']} shape_reject={round_stats['shape_reject']} "
            f"explain_error={round_stats['explain_error']} exec_error={round_stats['exec_error']}",
        )
        if round_stats["accepted"] == 0 and round_stats["returned"] > 0:
            fully_blocked = round_stats["quota_reject"] + round_stats["duplicate"] >= round_stats["returned"]
            nearly_blocked = round_stats["quota_reject"] >= max(5, round_stats["returned"] - 1)
            if fully_blocked or nearly_blocked:
                break
    for refine_seed_state in refine_queue:
        if len(runtime_state.accepted_results) >= settings.target_count:
            break
        expand_one_seed(
            settings=settings,
            runtime_state=runtime_state,
            seed_state=refine_seed_state,
            seed_index=seed_index,
        )
    return len(runtime_state.accepted_results) - accepted_before


def run_expander(settings: ExpanderSettings) -> dict:

    seed_sqls = load_seed_sqls(settings.seed_file)
    if not seed_sqls:
        raise RuntimeError(f"No seed SQL found in {settings.seed_file}")

    runtime_state = build_runtime_state(
        settings,
        seed_sqls,
        reset_outputs=True,
        diversity_source_count=len(seed_sqls),
    )
    try:
        log_line(settings.log_file, f"seed_count={len(seed_sqls)} target_count={settings.target_count}")
        for seed_index, seed_sql in enumerate(seed_sqls, start=1):
            if len(runtime_state.accepted_results) >= settings.target_count:
                break
            expand_one_seed(
                settings=settings,
                runtime_state=runtime_state,
                seed_state=SeedState(seed_sql=seed_sql),
                seed_index=seed_index,
            )
    finally:
        close_runtime_state(runtime_state)

    log_line(
        settings.log_file,
        f"finished accepted_count={len(runtime_state.accepted_results)} output_file={settings.output_file}",
    )
    return {
        "accepted_count": len(runtime_state.accepted_results),
        "seed_count": len(seed_sqls),
        "summary_file": str(settings.summary_file),
        "output_file": str(settings.output_file),
        "records_file": str(settings.records_file),
        "log_file": str(settings.log_file),
        "timing_log_file": str(settings.timing_log_file),
    }
