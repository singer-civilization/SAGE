from __future__ import annotations

import re
from dataclasses import asdict, dataclass


CLAUSE_KEYWORDS = {
    "where": " where ",
    "group_by": " group by ",
    "having": " having ",
    "order_by": " order by ",
}


@dataclass(frozen=True)
class PathConstraint:

    path_id: str
    path_name: str
    allowed_tables: tuple[str, ...]
    allowed_join_paths: tuple[tuple[str, str], ...]
    primary_tables: tuple[str, ...]
    secondary_tables: tuple[str, ...]
    from_mode: str
    required_predicate: bool
    allowed_clauses: tuple[str, ...]
    preferred_predicate_columns: tuple[str, ...]
    preferred_group_columns: tuple[str, ...]
    preferred_order_columns: tuple[str, ...]
    allow_aggregate: bool
    path_reason: str
    probe_sql: str = ""
    probe_runtime_seconds: float = -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "path_name": self.path_name,
            "allowed_tables": list(self.allowed_tables),
            "allowed_join_paths": [list(edge) for edge in self.allowed_join_paths],
            "primary_tables": list(self.primary_tables),
            "secondary_tables": list(self.secondary_tables),
            "from_mode": self.from_mode,
            "required_predicate": self.required_predicate,
            "allowed_clauses": list(self.allowed_clauses),
            "preferred_predicate_columns": list(self.preferred_predicate_columns),
            "preferred_group_columns": list(self.preferred_group_columns),
            "preferred_order_columns": list(self.preferred_order_columns),
            "allow_aggregate": self.allow_aggregate,
            "path_reason": self.path_reason,
        }


def normalize_sql(sql: str) -> str:

    normalized = " ".join(sql.strip().split()).lower()
    if not normalized.endswith(";"):
        normalized += ";"
    return normalized


def extract_sql_tables(sql: str, summary: dict) -> list[str]:

    normalized = normalize_sql(sql)
    tables = []
    for table_name in summary["tables"].keys():
        lowered = table_name.lower()
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(lowered)}(?![a-zA-Z0-9_])"
        if re.search(pattern, normalized):
            tables.append(table_name)
    return sorted(set(tables))


def sql_contains_clause(sql: str, clause_name: str) -> bool:

    keyword = CLAUSE_KEYWORDS[clause_name]
    return keyword in normalize_sql(sql)


def sql_uses_aggregate(sql: str) -> bool:

    normalized = normalize_sql(sql)
    return any(token in normalized for token in (" count(", " max(", " min(", " avg(", " sum("))


def sql_matches_constraint(sql: str, summary: dict, constraint: PathConstraint) -> tuple[bool, str]:

    normalized = normalize_sql(sql)
    tables = extract_sql_tables(normalized, summary)
    if not tables:
        return False, "No table could be parsed"

    if not set(tables).issubset(set(constraint.allowed_tables)):
        return False, "SQL uses tables outside the allowed path scope"

    if constraint.from_mode == "single" and len(tables) != 1:
        return False, "Path requires a single table"

    if constraint.from_mode == "join":
        if len(tables) < 2 or " join " not in normalized:
            return False, "Path requires an explicit join"

    if constraint.required_predicate and not (
        sql_contains_clause(normalized, "where") or sql_contains_clause(normalized, "having")
    ):
        return False, "Missing predicate"

    for clause_name in CLAUSE_KEYWORDS:
        if sql_contains_clause(normalized, clause_name) and clause_name not in constraint.allowed_clauses:
            return False, f"Clause {clause_name} is not allowed"

    if sql_uses_aggregate(normalized) and not constraint.allow_aggregate:
        return False, "Path does not allow aggregation"

    return True, "ok"
