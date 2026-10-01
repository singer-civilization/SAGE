import json

from agent.expander.config import RuntimeWindow


TARGET_CAUSES = [
    (
        "index_missing_or_inappropriate",
        "Missing or unsuitable indexes. SQL should include WHERE/HAVING predicates so the plan tends toward Seq Scan or inefficient filtering over larger data, rather than using a suitable index.",
    ),
    (
        "inefficient_predicate_or_complex_subquery",
        "Inefficient predicates or complex subqueries. SQL should include inefficient predicates such as LIKE/OR/NOT/function expressions, or subquery structures such as EXISTS/IN/nested SELECT. Keep the query realistic instead of inventing impractical complexity.",
    ),
    (
        "high_sort_aggregation_join_cost",
        "High sort, aggregation, or join cost. SQL should explicitly include ORDER BY, GROUP BY/aggregation/HAVING, or multi-table joins, with slowness caused by these operators.",
    ),
]


def build_system_prompt() -> str:

    return (
        "You are a PostgreSQL SQL workload expander. "
        "Your task is to generate executable, structurally diverse SELECT SQL with clear slow-cause mechanisms and runtimes close to the target, using the seed SQL and database summary. "
        "Output compact JSON only. Do not output explanations, Markdown, code fences, indentation, line breaks, or extra spaces."
    )


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _compact_path_context(path_context: dict | None) -> dict:
    if path_context is None:
        return {}
    hard_keys = [
        "path_id",
        "path_name",
        "allowed_tables",
        "allowed_join_paths",
        "from_mode",
        "required_predicate",
        "allowed_clauses",
        "allow_aggregate",
        "preferred_predicate_columns",
        "preferred_group_columns",
        "preferred_order_columns",
    ]
    return {key: path_context[key] for key in hard_keys if key in path_context}


def _shorten_recent_failures(recent_failures: list[str]) -> list[str]:
    return [failure[:300] for failure in recent_failures[-2:]]


def _build_join_key_hints(related_summary: dict, path_context: dict | None) -> list[dict[str, str]]:

    if not path_context:
        return []
    allowed_join_paths = {
        tuple(sorted(edge))
        for edge in path_context.get("allowed_join_paths", [])
        if len(edge) == 2
    }
    if not allowed_join_paths:
        return []

    tables = related_summary.get("tables", {})
    hints: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for table_name, table_info in tables.items():
        for foreign_key in table_info.get("foreign_keys", []):
            referenced_table = foreign_key["referenced_table"]
            edge = tuple(sorted((table_name, referenced_table)))
            if edge not in allowed_join_paths:
                continue
            key = (
                table_name,
                foreign_key["column"],
                referenced_table,
                foreign_key["referenced_column"],
            )
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "left_table": table_name,
                    "left_column": f"{table_name}.{foreign_key['column']}",
                    "right_table": referenced_table,
                    "right_column": f"{referenced_table}.{foreign_key['referenced_column']}",
                    "condition": (
                        f"{table_name}.{foreign_key['column']} = "
                        f"{referenced_table}.{foreign_key['referenced_column']}"
                    ),
                    "source": foreign_key.get("source", "database"),
                }
            )
    return hints


def build_multi_direction_user_prompt(
    seed_sql: str,
    related_summary: dict,
    runtime_window: RuntimeWindow,
    total_candidate_limit: int,
    recent_failures: list[str],
    recent_success_fingerprints: list[str],
    path_context: dict | None,
    refine_depth: int,
) -> str:

    target_causes = [{"name": name, "description": description} for name, description in TARGET_CAUSES]
    runtime_target = {
        "target_seconds": round(runtime_window.target_seconds, 3),
        "accept_min_seconds": round(runtime_window.accept_min_seconds, 3),
        "accept_max_seconds": round(runtime_window.accept_max_seconds, 3),
        "slow_accept_max_seconds": round(runtime_window.slow_accept_max_seconds, 3),
        "timeout_seconds": round(runtime_window.timeout_seconds, 3),
    }
    feedback = {
        "recent_failures": _shorten_recent_failures(recent_failures),
        "accepted_fingerprints": recent_success_fingerprints[-3:],
    }
    join_key_hints = _build_join_key_hints(related_summary, path_context)
    refine_guidance = (
        "The current seed is verified stable but faster than the target. "
        "Prioritize rewrites that increase execution cost, such as loosening predicates, adding sorting, adding GROUP BY/HAVING, "
        "expanding join fan-out, or widening projections, while staying on the business path."
        if refine_depth > 0
        else "The current seed is the initial expansion point, so explore both faster and slower directions."
    )
    return f"""
Expand based on the seed SQL below.
The seed SQL is only a reference for the business path, table-column relationships, and rough complexity. Do not treat the task as a local rewrite.

seed SQL:
{seed_sql}

Runtime target:
{_compact_json(runtime_target)}

Available target slow causes:
{_compact_json(target_causes)}

Database summary:
{_compact_json(related_summary)}

Business path hard constraints:
{_compact_json(_compact_path_context(path_context))}

Available join keys:
{_compact_json(join_key_hints)}

Expansion strategy for this round:
refine_depth={refine_depth};{refine_guidance}

Feedback:
{_compact_json(feedback)}

Hard requirements:
1. Generate PostgreSQL SELECT SQL only. Every statement must end with a semicolon.
2. target_cause must be chosen from the name field of the available target slow causes.
3. If business path hard constraints are non-empty, use only tables from allowed_tables.
4. If from_mode=join, SQL must explicitly use JOIN. Joined table pairs must follow allowed_join_paths, and ON conditions should prefer the condition values from available join keys.
5. If required_predicate=true, SQL must include WHERE or HAVING.
6. Use only WHERE/GROUP BY/HAVING/ORDER BY clauses allowed by allowed_clauses.
7. If allow_aggregate=false, do not generate aggregate functions, GROUP BY, or HAVING.
8. Do not return a simple rewrite of the seed SQL. The query must show the slow-cause mechanism for target_cause, not just replace constants.
9. Prefer stable candidates likely to fall between accept_min_seconds and slow_accept_max_seconds.
10. Return at most {total_candidate_limit} candidates and cover different target_cause values where possible.
11. Do not output explanations. Do not format or indent. Return compact JSON.
12. change_summary must explain the slow-cause mechanism, such as scan filtering due to missing indexes, inefficient predicates / complex subqueries, or sort / aggregation / join cost.
13. The seed SQL is only a path reference. Do not reuse the seed query skeleton; the skeleton includes SELECT projection columns, WHERE/HAVING predicate columns, and GROUP/ORDER columns.
14. Candidates in the same round must use different query_shape values, and main_projection_columns plus main_predicate_columns must not be identical.
15. Rotate query perspectives where possible: dimension_view, fact_detail, aggregation_view, sorted_wide_scan, subquery_filter.
16. If recent failures mention shape_reject, the next round must change projection columns, predicate columns, or GROUP/ORDER columns instead of only changing constants.

Return format must be exactly:
{{"candidates":[{{"sql":"select ...;","target_cause":"index_missing_or_inappropriate","query_shape":"fact_detail","main_projection_columns":["partsupp.ps_partkey"],"main_predicate_columns":["partsupp.ps_comment"],"main_group_order_columns":["partsupp.ps_partkey"],"change_summary":"Explain the slow-cause mechanism and skeleton change for this SQL."}}]}}
""".strip()
