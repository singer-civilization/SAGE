import re
import json
import statistics
import time
from dataclasses import dataclass


@dataclass
class ValidationResult:

    sql: str
    fingerprint: str
    status: str
    runtime_seconds: float
    accepted: bool
    message: str


@dataclass
class StableValidationResult:

    sql: str
    fingerprint: str
    status: str
    runtime_seconds: float
    runtime_min: float
    runtime_max: float
    runtime_std: float
    runtime_band: str
    accepted: bool
    message: str
    run_count: int


def normalize_sql(sql: str) -> str:

    compact = " ".join(sql.strip().split())
    if not compact.endswith(";"):
        compact += ";"
    return compact


def is_select_sql(sql: str) -> bool:

    return normalize_sql(sql).lower().startswith("select ")


def structural_fingerprint(sql: str) -> str:

    normalized = normalize_sql(sql).lower()
    normalized = re.sub(r"'([^']|'')*'", "?", normalized)
    normalized = re.sub(r"\b\d+(\.\d+)?\b", "?", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_execution_time_ms(plan_payload: object) -> float:

    if isinstance(plan_payload, list) and plan_payload:
        root = plan_payload[0]
    else:
        root = plan_payload
    if isinstance(root, dict):
        execution_time = root.get("Execution Time")
        if isinstance(execution_time, (int, float)):
            return float(execution_time)
    raise ValueError("EXPLAIN ANALYZE output does not contain Execution Time")


def execute_explain_analyze(connection, sql: str, timeout_seconds: float) -> tuple[str, float, str]:

    timeout_ms = int(timeout_seconds * 1000)
    try:
        connection.rollback()
        connection.autocommit = False
        with connection.cursor() as cursor:
            cursor.execute(f"SET LOCAL statement_timeout = {timeout_ms};")
            cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
            row = cursor.fetchone()
        plan_payload = row[0]
        if isinstance(plan_payload, str):
            plan_payload = json.loads(plan_payload)
        runtime_seconds = extract_execution_time_ms(plan_payload) / 1000.0
        connection.commit()
        return "ok", runtime_seconds, "ok"
    except Exception as exc:
        connection.rollback()
        message = str(exc)
        if "canceling statement due to statement timeout" in message:
            return "timeout", timeout_seconds, message
        return "error", 0.0, message


def explain_sql(connection, sql: str) -> tuple[bool, str]:

    try:
        with connection.cursor() as cursor:
            cursor.execute("EXPLAIN " + sql)
            cursor.fetchall()
        return True, "ok"
    except Exception as exc:
        connection.rollback()
        return False, str(exc)


def explain_plan_json(connection, sql: str) -> tuple[bool, object, str]:

    try:
        with connection.cursor() as cursor:
            cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
            row = cursor.fetchone()
        plan_payload = row[0]
        if isinstance(plan_payload, str):
            plan_payload = json.loads(plan_payload)
        return True, plan_payload, "ok"
    except Exception as exc:
        connection.rollback()
        return False, {}, str(exc)


def execute_sql(connection, sql: str, timeout_seconds: float) -> tuple[str, float, str]:

    return execute_explain_analyze(connection, sql, timeout_seconds)


def classify_runtime_band(runtime_seconds: float, runtime_window) -> str:

    if runtime_seconds < runtime_window.accept_min_seconds:
        return "fast"
    if runtime_seconds <= runtime_window.accept_max_seconds:
        return "near_target"
    if runtime_seconds < runtime_window.timeout_seconds:
        return "slow"
    return "timeout"


def validate_candidate(connection, sql: str, runtime_window) -> ValidationResult:

    normalized_sql = normalize_sql(sql)
    fingerprint = structural_fingerprint(normalized_sql)

    if not is_select_sql(normalized_sql):
        return ValidationResult(
            sql=normalized_sql,
            fingerprint=fingerprint,
            status="non_select",
            runtime_seconds=0.0,
            accepted=False,
            message="Only SELECT SQL is allowed",
        )

    explain_ok, explain_message = explain_sql(connection, normalized_sql)
    if not explain_ok:
        return ValidationResult(
            sql=normalized_sql,
            fingerprint=fingerprint,
            status="explain_error",
            runtime_seconds=0.0,
            accepted=False,
            message=explain_message,
        )

    status, runtime_seconds, message = execute_sql(
        connection=connection,
        sql=normalized_sql,
        timeout_seconds=runtime_window.timeout_seconds,
    )
    accepted = (
        status == "ok"
        and runtime_window.accept_min_seconds <= runtime_seconds <= runtime_window.accept_max_seconds
    )
    return ValidationResult(
        sql=normalized_sql,
        fingerprint=fingerprint,
        status=status,
        runtime_seconds=runtime_seconds,
        accepted=accepted,
        message=message,
    )


def validate_candidate_stable(
    connection,
    sql: str,
    runtime_window,
    run_count: int,
    max_runtime_cv: float,
    max_runtime_spread: float,
) -> StableValidationResult:

    normalized_sql = normalize_sql(sql)
    fingerprint = structural_fingerprint(normalized_sql)

    if not is_select_sql(normalized_sql):
        return StableValidationResult(
            sql=normalized_sql,
            fingerprint=fingerprint,
            status="non_select",
            runtime_seconds=0.0,
            runtime_min=0.0,
            runtime_max=0.0,
            runtime_std=0.0,
            runtime_band="invalid",
            accepted=False,
            message="Only SELECT SQL is allowed",
            run_count=0,
        )

    explain_ok, explain_message = explain_sql(connection, normalized_sql)
    if not explain_ok:
        return StableValidationResult(
            sql=normalized_sql,
            fingerprint=fingerprint,
            status="explain_error",
            runtime_seconds=0.0,
            runtime_min=0.0,
            runtime_max=0.0,
            runtime_std=0.0,
            runtime_band="invalid",
            accepted=False,
            message=explain_message,
            run_count=0,
        )

    runtimes: list[float] = []
    for _ in range(run_count):
        status, runtime_seconds, message = execute_sql(
            connection=connection,
            sql=normalized_sql,
            timeout_seconds=runtime_window.timeout_seconds,
        )
        if status != "ok":
            runtime_band = classify_runtime_band(runtime_seconds, runtime_window)
            return StableValidationResult(
                sql=normalized_sql,
                fingerprint=fingerprint,
                status=status,
                runtime_seconds=runtime_seconds,
                runtime_min=runtime_seconds,
                runtime_max=runtime_seconds,
                runtime_std=0.0,
                runtime_band=runtime_band,
                accepted=False,
                message=message,
                run_count=len(runtimes) + 1,
            )
        runtimes.append(runtime_seconds)

    runtime_mean = sum(runtimes) / len(runtimes)
    runtime_min = min(runtimes)
    runtime_max = max(runtimes)
    runtime_std = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
    runtime_cv = runtime_std / runtime_mean if runtime_mean > 0 else float("inf")
    runtime_spread = runtime_max / runtime_min if runtime_min > 0 else float("inf")
    runtime_band = classify_runtime_band(runtime_mean, runtime_window)
    if runtime_mean >= runtime_window.timeout_seconds:
        return StableValidationResult(
            sql=normalized_sql,
            fingerprint=fingerprint,
            status="over_budget",
            runtime_seconds=runtime_mean,
            runtime_min=runtime_min,
            runtime_max=runtime_max,
            runtime_std=runtime_std,
            runtime_band="timeout",
            accepted=False,
            message=f"runtime_mean={runtime_mean:.6f}s exceeded timeout budget",
            run_count=len(runtimes),
        )

    is_stable = runtime_cv <= max_runtime_cv and runtime_spread <= max_runtime_spread
    status = "ok" if is_stable else "unstable"
    message = (
        "ok"
        if is_stable
        else f"unstable cv={runtime_cv:.4f} spread={runtime_spread:.4f}"
    )
    return StableValidationResult(
        sql=normalized_sql,
        fingerprint=fingerprint,
        status=status,
        runtime_seconds=runtime_mean,
        runtime_min=runtime_min,
        runtime_max=runtime_max,
        runtime_std=runtime_std,
        runtime_band=runtime_band,
        accepted=is_stable,
        message=message,
        run_count=len(runtimes),
    )
