from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.planner.models import PathConstraint
from pool import load_or_build_summary


@dataclass(frozen=True)
class PlanBundle:

    summary: dict
    paths: tuple[PathConstraint, ...]
    candidate_count: int
    candidate_bucket_counts: dict[str, int]


def build_fk_graph(summary: dict) -> dict[str, set[str]]:

    graph: dict[str, set[str]] = {table_name: set() for table_name in summary["tables"].keys()}
    for table_name, table_info in summary["tables"].items():
        for foreign_key in table_info["foreign_keys"]:
            referenced_table = foreign_key["referenced_table"]
            graph[table_name].add(referenced_table)
            graph.setdefault(referenced_table, set()).add(table_name)
    return graph


def pick_columns(table_info: dict, *, kinds: tuple[str, ...], limit: int) -> tuple[str, ...]:

    selected: list[str] = []
    for column in table_info["columns"]:
        if column["data_type"] in kinds:
            selected.append(column["name"])
        if len(selected) >= limit:
            break
    return tuple(selected)


def build_single_table_path(summary: dict, table_name: str) -> PathConstraint:

    table_info = summary["tables"][table_name]
    predicate_columns = pick_columns(
        table_info,
        kinds=("integer", "numeric", "date", "character varying", "text", "character"),
        limit=4,
    )
    group_columns = pick_columns(
        table_info,
        kinds=("character varying", "text", "character", "date"),
        limit=3,
    )
    order_columns = pick_columns(
        table_info,
        kinds=("integer", "numeric", "date", "character varying", "text", "character"),
        limit=3,
    )
    return PathConstraint(
        path_id=f"path_single_{table_name}",
        path_name=f"{table_name} single-table path",
        allowed_tables=(table_name,),
        allowed_join_paths=(),
        primary_tables=(table_name,),
        secondary_tables=(),
        from_mode="single",
        required_predicate=False,
        allowed_clauses=("where", "order_by", "group_by", "having"),
        preferred_predicate_columns=tuple(f"{table_name}.{name}" for name in predicate_columns),
        preferred_group_columns=tuple(f"{table_name}.{name}" for name in group_columns),
        preferred_order_columns=tuple(f"{table_name}.{name}" for name in order_columns),
        allow_aggregate=True,
        path_reason="Generated from a single table, suitable for stable seed SQL with filters and ordering.",
    )


def build_join_path(summary: dict, table_names: tuple[str, ...]) -> PathConstraint:

    edge_set: set[tuple[str, str]] = set()
    for table_name in table_names:
        table_info = summary["tables"][table_name]
        for foreign_key in table_info["foreign_keys"]:
            referenced_table = foreign_key["referenced_table"]
            if referenced_table in table_names:
                edge_set.add(tuple(sorted((table_name, referenced_table))))

    primary_table = table_names[0]
    primary_info = summary["tables"][primary_table]
    predicate_columns = pick_columns(
        primary_info,
        kinds=("integer", "numeric", "date", "character varying", "text", "character"),
        limit=4,
    )
    group_columns = pick_columns(
        primary_info,
        kinds=("character varying", "text", "character", "date"),
        limit=3,
    )
    order_columns = pick_columns(
        primary_info,
        kinds=("integer", "numeric", "date", "character varying", "text", "character"),
        limit=3,
    )

    return PathConstraint(
        path_id="path_join_" + "_".join(table_names),
        path_name=" -> ".join(table_names),
        allowed_tables=table_names,
        allowed_join_paths=tuple(sorted(edge_set)),
        primary_tables=(primary_table,),
        secondary_tables=tuple(table_names[1:]),
        from_mode="join",
        required_predicate=False,
        allowed_clauses=("where", "order_by", "group_by", "having"),
        preferred_predicate_columns=tuple(f"{primary_table}.{name}" for name in predicate_columns),
        preferred_group_columns=tuple(f"{primary_table}.{name}" for name in group_columns),
        preferred_order_columns=tuple(f"{primary_table}.{name}" for name in order_columns),
        allow_aggregate=True,
        path_reason="Generated from foreign-key links, prioritizing natural join paths for workload construction.",
    )


def enumerate_paths(summary: dict) -> list[PathConstraint]:

    graph = build_fk_graph(summary)
    constraints: list[PathConstraint] = []
    seen: set[tuple[str, ...]] = set()

    for table_name in sorted(summary["tables"].keys()):
        constraints.append(build_single_table_path(summary, table_name))

    for table_name in sorted(graph.keys()):
        for neighbor in sorted(graph[table_name]):
            pair = tuple(sorted((table_name, neighbor)))
            if pair in seen:
                continue
            seen.add(pair)
            constraints.append(build_join_path(summary, pair))

    chain_seen: set[tuple[str, ...]] = set()
    for table_name in sorted(graph.keys()):
        neighbors = sorted(graph[table_name])
        for neighbor in neighbors:
            for second_neighbor in sorted(graph[neighbor]):
                if second_neighbor == table_name:
                    continue
                chain = (table_name, neighbor, second_neighbor)
                chain_key = tuple(dict.fromkeys(chain))
                if len(chain_key) < 3:
                    continue
                normalized_chain = tuple(chain_key)
                if normalized_chain in chain_seen:
                    continue
                chain_seen.add(normalized_chain)
                constraints.append(build_join_path(summary, normalized_chain))
    return constraints


def score_path(constraint: PathConstraint) -> tuple[int, int, int]:

    return (
        len(constraint.allowed_tables),
        len(constraint.allowed_join_paths),
        len(constraint.preferred_predicate_columns),
    )


def estimate_path_rows(summary: dict, constraint: PathConstraint) -> float:

    total = 0.0
    for table_name in constraint.allowed_tables:
        total += float(summary["tables"][table_name].get("row_estimate", 0.0))
    return total


def path_size_score(summary: dict, constraint: PathConstraint) -> tuple[int, float, tuple[int, int, int]]:

    row_estimate = estimate_path_rows(summary, constraint)
    table_count = len(constraint.allowed_tables)
    if row_estimate <= 0:
        size_bucket = 5
    elif 1_000_000 <= row_estimate <= 15_000_000:
        size_bucket = 0
    elif 300_000 <= row_estimate < 1_000_000:
        size_bucket = 1
    elif 15_000_000 < row_estimate <= 35_000_000:
        size_bucket = 2
    elif 100_000 <= row_estimate < 300_000:
        size_bucket = 3
    elif row_estimate > 35_000_000:
        size_bucket = 4
    else:
        size_bucket = 5
    return (size_bucket, row_estimate / max(table_count, 1), score_path(constraint))


def preselect_candidates(summary: dict, candidate_paths: list[PathConstraint]) -> list[PathConstraint]:

    singles = sorted(
        [path for path in candidate_paths if len(path.allowed_tables) == 1],
        key=lambda path: path_size_score(summary, path),
    )
    join_pairs = sorted(
        [path for path in candidate_paths if len(path.allowed_tables) == 2],
        key=lambda path: path_size_score(summary, path),
    )
    join_chains = sorted(
        [path for path in candidate_paths if len(path.allowed_tables) > 2],
        key=lambda path: path_size_score(summary, path),
    )
    return singles + join_pairs + join_chains


def table_combo_key(constraint: PathConstraint) -> tuple[str, ...]:

    return tuple(sorted(constraint.allowed_tables))


def bucket_name(constraint: PathConstraint) -> str:

    table_count = len(constraint.allowed_tables)
    if table_count == 1:
        return "single"
    if table_count == 2:
        return "pair"
    return "chain"


def count_buckets(paths: list[PathConstraint]) -> dict[str, int]:

    counts = {"single": 0, "pair": 0, "chain": 0}
    for path in paths:
        counts[bucket_name(path)] += 1
    return counts


def pop_bucket_candidate(
    bucket: list[PathConstraint],
    used_primary_tables: set[str],
) -> PathConstraint | None:

    if not bucket:
        return None
    for index, constraint in enumerate(bucket):
        primary_table = constraint.primary_tables[0] if constraint.primary_tables else ""
        if primary_table not in used_primary_tables:
            return bucket.pop(index)
    return bucket.pop(0)


def select_diverse_paths(ranked_paths: list[PathConstraint], max_paths: int) -> list[PathConstraint]:

    if max_paths <= 0:
        return []

    buckets = {"single": [], "pair": [], "chain": []}
    seen_combo_keys: set[tuple[str, ...]] = set()
    for constraint in ranked_paths:
        combo_key = table_combo_key(constraint)
        if combo_key in seen_combo_keys:
            continue
        seen_combo_keys.add(combo_key)
        buckets[bucket_name(constraint)].append(constraint)

    selected: list[PathConstraint] = []
    used_primary_tables: set[str] = set()
    round_robin_order = ("single", "pair", "chain")
    while len(selected) < max_paths:
        progress = False
        for bucket_key in round_robin_order:
            candidate = pop_bucket_candidate(buckets[bucket_key], used_primary_tables)
            if candidate is None:
                continue
            selected.append(candidate)
            if candidate.primary_tables:
                used_primary_tables.add(candidate.primary_tables[0])
            progress = True
            if len(selected) >= max_paths:
                break
        if not progress:
            break
    return selected


def extend_with_diverse_paths(
    selected_paths: list[PathConstraint],
    candidate_paths: list[PathConstraint],
    max_paths: int,
) -> list[PathConstraint]:

    if len(selected_paths) >= max_paths:
        return selected_paths
    used_path_ids = {path.path_id for path in selected_paths}
    remaining = [path for path in candidate_paths if path.path_id not in used_path_ids]
    selected_paths.extend(select_diverse_paths(remaining, max_paths - len(selected_paths)))
    return selected_paths


def save_plan_outputs(
    paths: list[PathConstraint],
    paths_file: Path,
    constraints_file: Path,
) -> None:

    paths_file.parent.mkdir(parents=True, exist_ok=True)
    constraints_file.parent.mkdir(parents=True, exist_ok=True)
    path_dicts = [path.to_dict() for path in paths]
    paths_file.write_text(json.dumps(path_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    constraints_file.write_text(json.dumps(path_dicts, ensure_ascii=False, indent=2), encoding="utf-8")


def build_plan_bundle(summary_file: Path, sample_value_limit: int, max_paths: int, runtime_window) -> PlanBundle:

    summary = load_or_build_summary(summary_file, sample_value_limit)
    candidate_paths = enumerate_paths(summary)
    ranked_paths = preselect_candidates(summary, candidate_paths)
    medium_paths = [path for path in ranked_paths if estimate_path_rows(summary, path) <= 15_000_000]
    large_paths = [
        path
        for path in ranked_paths
        if 15_000_000 < estimate_path_rows(summary, path) <= 35_000_000
    ]
    huge_paths = [path for path in ranked_paths if estimate_path_rows(summary, path) > 35_000_000]
    selected_paths: list[PathConstraint] = []
    selected_paths = extend_with_diverse_paths(selected_paths, medium_paths, max_paths)
    selected_paths = extend_with_diverse_paths(selected_paths, large_paths, max_paths)
    selected_paths = extend_with_diverse_paths(selected_paths, huge_paths, max_paths)
    return PlanBundle(
        summary=summary,
        paths=tuple(selected_paths),
        candidate_count=len(candidate_paths),
        candidate_bucket_counts=count_buckets(candidate_paths),
    )
