import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)


def require_env(name: str) -> str:

    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_path(raw_path: str) -> Path:

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def format_seconds(value: float) -> str:

    return str(float(value))


def render_path_template(
    raw_path: str,
    *,
    database: str,
    seed_target_seconds: float,
    expander_target_seconds: float,
) -> str:

    return raw_path.format(
        database=database,
        dbname=database,
        seed_target=format_seconds(seed_target_seconds),
        expander_target=format_seconds(expander_target_seconds),
    )


def resolve_artifact_path(
    raw_path: str,
    *,
    database: str,
    seed_target_seconds: float,
    expander_target_seconds: float,
) -> Path:

    return resolve_path(
        render_path_template(
            raw_path,
            database=database,
            seed_target_seconds=seed_target_seconds,
            expander_target_seconds=expander_target_seconds,
        )
    )


@dataclass(frozen=True)
class RuntimeWindow:

    target_seconds: float
    accept_min_seconds: float
    accept_max_seconds: float
    slow_accept_max_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class DatabaseSettings:

    host: str
    port: str
    user: str
    password: str
    database: str
    min_conn: int
    max_conn: int


@dataclass(frozen=True)
class SeedGeneratorSettings:

    dbname: str
    sql_count: int
    target_seconds: float
    accept_min_ratio: float
    accept_max_ratio: float
    timeout_ratio: float
    from_mode: str
    max_steps: int


@dataclass(frozen=True)
class ExpanderSettings:

    database: DatabaseSettings
    seed_generator: SeedGeneratorSettings
    api_key: str
    base_url: str
    model: str
    seed_file: Path
    summary_file: Path
    output_file: Path
    records_file: Path
    log_file: Path
    timing_log_file: Path
    checkpoint_file: Path
    resume_run: bool
    planner_paths_file: Path
    planner_constraints_file: Path
    target_count: int
    candidates_per_request: int
    max_rounds_per_seed: int
    request_timeout_seconds: int
    sample_value_limit: int
    planner_max_paths: int
    live_seeds_per_turn: int
    live_max_seeds_per_path: int
    live_generator_attempts_per_turn: int
    live_max_no_gain_turns: int
    validate_runs: int
    max_runtime_cv: float
    max_runtime_spread: float
    runtime_window: RuntimeWindow


def build_runtime_window(
    target_seconds: float,
    accept_min_ratio: float,
    accept_max_ratio: float,
    slow_accept_max_ratio: float,
    timeout_ratio: float,
) -> RuntimeWindow:

    accept_min_seconds = target_seconds * accept_min_ratio
    accept_max_seconds = target_seconds * accept_max_ratio
    slow_accept_max_seconds = target_seconds * slow_accept_max_ratio
    timeout_seconds = target_seconds * timeout_ratio
    return RuntimeWindow(
        target_seconds=target_seconds,
        accept_min_seconds=accept_min_seconds,
        accept_max_seconds=accept_max_seconds,
        slow_accept_max_seconds=slow_accept_max_seconds,
        timeout_seconds=timeout_seconds,
    )


def parse_bool_env(name: str) -> bool:

    raw_value = require_env(name).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def load_settings() -> ExpanderSettings:

    database = DatabaseSettings(
        host=require_env("PGHOST"),
        port=require_env("PGPORT"),
        user=require_env("PGUSER"),
        password=require_env("PGPASSWORD"),
        database=require_env("PGDATABASE"),
        min_conn=int(require_env("PG_MIN_CONN")),
        max_conn=int(require_env("PG_MAX_CONN")),
    )
    if database.min_conn <= 0:
        raise RuntimeError("PG_MIN_CONN must be greater than 0")
    if database.max_conn < database.min_conn:
        raise RuntimeError("PG_MAX_CONN must be greater than or equal to PG_MIN_CONN")

    seed_generator = SeedGeneratorSettings(
        dbname=database.database,
        sql_count=int(require_env("GEN_SLOW_SQL_COUNT")),
        target_seconds=float(require_env("GEN_SLOW_SQL_TARGET_SECONDS")),
        accept_min_ratio=float(require_env("GEN_SLOW_SQL_ACCEPT_MIN_RATIO")),
        accept_max_ratio=float(require_env("GEN_SLOW_SQL_ACCEPT_MAX_RATIO")),
        timeout_ratio=float(require_env("GEN_SLOW_SQL_TIMEOUT_RATIO")),
        from_mode=require_env("GEN_SLOW_SQL_FROM_MODE"),
        max_steps=int(require_env("GEN_SLOW_SQL_MAX_STEPS")),
    )
    if seed_generator.sql_count <= 0:
        raise RuntimeError("GEN_SLOW_SQL_COUNT must be greater than 0")
    if seed_generator.target_seconds <= 0:
        raise RuntimeError("GEN_SLOW_SQL_TARGET_SECONDS must be greater than 0")
    if seed_generator.accept_min_ratio >= seed_generator.accept_max_ratio:
        raise RuntimeError("GEN_SLOW_SQL_ACCEPT_MIN_RATIO must be less than GEN_SLOW_SQL_ACCEPT_MAX_RATIO")
    if seed_generator.timeout_ratio <= 0:
        raise RuntimeError("GEN_SLOW_SQL_TIMEOUT_RATIO must be greater than 0")
    if seed_generator.from_mode not in {"single", "join", "cartesian"}:
        raise RuntimeError("GEN_SLOW_SQL_FROM_MODE must be single, join, or cartesian")
    if seed_generator.max_steps <= 0:
        raise RuntimeError("GEN_SLOW_SQL_MAX_STEPS must be greater than 0")

    runtime_window = build_runtime_window(
        target_seconds=float(require_env("EXPANDER_TARGET_SECONDS")),
        accept_min_ratio=float(require_env("EXPANDER_ACCEPT_MIN_RATIO")),
        accept_max_ratio=float(require_env("EXPANDER_ACCEPT_MAX_RATIO")),
        slow_accept_max_ratio=float(require_env("EXPANDER_SLOW_ACCEPT_MAX_RATIO")),
        timeout_ratio=float(require_env("EXPANDER_TIMEOUT_RATIO")),
    )
    if runtime_window.accept_min_seconds >= runtime_window.accept_max_seconds:
        raise RuntimeError("EXPANDER_ACCEPT_MIN_RATIO must be less than EXPANDER_ACCEPT_MAX_RATIO")
    if runtime_window.slow_accept_max_seconds < runtime_window.accept_max_seconds:
        raise RuntimeError("EXPANDER_SLOW_ACCEPT_MAX_RATIO must be greater than or equal to EXPANDER_ACCEPT_MAX_RATIO")
    if runtime_window.slow_accept_max_seconds > runtime_window.timeout_seconds:
        raise RuntimeError("EXPANDER_SLOW_ACCEPT_MAX_RATIO must be less than or equal to EXPANDER_TIMEOUT_RATIO")
    if runtime_window.timeout_seconds <= 0:
        raise RuntimeError("EXPANDER_TIMEOUT_RATIO must be greater than 0")

    target_count = int(require_env("EXPANDER_TARGET_COUNT"))
    candidates_per_request = int(require_env("EXPANDER_CANDIDATES_PER_REQUEST"))
    max_rounds_per_seed = int(require_env("EXPANDER_MAX_ROUNDS_PER_SEED"))
    request_timeout_seconds = int(require_env("EXPANDER_REQUEST_TIMEOUT_SECONDS"))
    sample_value_limit = int(require_env("EXPANDER_SAMPLE_VALUE_LIMIT"))
    planner_max_paths = int(require_env("PLANNER_MAX_PATHS"))
    live_seeds_per_turn = int(require_env("LIVE_SEEDS_PER_TURN"))
    live_max_seeds_per_path = int(require_env("LIVE_MAX_SEEDS_PER_PATH"))
    live_generator_attempts_per_turn = int(require_env("LIVE_GENERATOR_ATTEMPTS_PER_TURN"))
    live_max_no_gain_turns = int(require_env("LIVE_MAX_NO_GAIN_TURNS"))
    validate_runs = int(require_env("EXPANDER_VALIDATE_RUNS"))
    max_runtime_cv = float(require_env("EXPANDER_MAX_RUNTIME_CV"))
    max_runtime_spread = float(require_env("EXPANDER_MAX_RUNTIME_SPREAD"))
    if target_count <= 0:
        raise RuntimeError("EXPANDER_TARGET_COUNT must be greater than 0")
    if candidates_per_request <= 0:
        raise RuntimeError("EXPANDER_CANDIDATES_PER_REQUEST must be greater than 0")
    if max_rounds_per_seed <= 0:
        raise RuntimeError("EXPANDER_MAX_ROUNDS_PER_SEED must be greater than 0")
    if request_timeout_seconds <= 0:
        raise RuntimeError("EXPANDER_REQUEST_TIMEOUT_SECONDS must be greater than 0")
    if sample_value_limit <= 0:
        raise RuntimeError("EXPANDER_SAMPLE_VALUE_LIMIT must be greater than 0")
    if planner_max_paths <= 0:
        raise RuntimeError("PLANNER_MAX_PATHS must be greater than 0")
    if live_seeds_per_turn <= 0:
        raise RuntimeError("LIVE_SEEDS_PER_TURN must be greater than 0")
    if live_max_seeds_per_path <= 0:
        raise RuntimeError("LIVE_MAX_SEEDS_PER_PATH must be greater than 0")
    if live_generator_attempts_per_turn <= 0:
        raise RuntimeError("LIVE_GENERATOR_ATTEMPTS_PER_TURN must be greater than 0")
    if live_max_no_gain_turns <= 0:
        raise RuntimeError("LIVE_MAX_NO_GAIN_TURNS must be greater than 0")
    if validate_runs < 1:
        raise RuntimeError("EXPANDER_VALIDATE_RUNS must be greater than or equal to 1")
    if max_runtime_cv <= 0:
        raise RuntimeError("EXPANDER_MAX_RUNTIME_CV must be greater than 0")
    if max_runtime_spread < 1.0:
        raise RuntimeError("EXPANDER_MAX_RUNTIME_SPREAD must be greater than or equal to 1.0")

    settings = ExpanderSettings(
        database=database,
        seed_generator=seed_generator,
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=require_env("DEEPSEEK_BASE_URL").rstrip("/"),
        model=require_env("DEEPSEEK_MODEL"),
        seed_file=resolve_artifact_path(
            require_env("EXPANDER_SEED_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        summary_file=resolve_artifact_path(
            require_env("EXPANDER_SUMMARY_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        output_file=resolve_artifact_path(
            require_env("EXPANDER_OUTPUT_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        records_file=resolve_artifact_path(
            require_env("EXPANDER_RECORDS_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        log_file=resolve_artifact_path(
            require_env("EXPANDER_RUN_LOG"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        timing_log_file=resolve_artifact_path(
            require_env("EXPANDER_TIMING_LOG"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        checkpoint_file=resolve_artifact_path(
            require_env("EXPANDER_CHECKPOINT_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        resume_run=parse_bool_env("EXPANDER_RESUME_RUN"),
        planner_paths_file=resolve_artifact_path(
            require_env("PLANNER_PATHS_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        planner_constraints_file=resolve_artifact_path(
            require_env("PLANNER_CONSTRAINTS_FILE"),
            database=database.database,
            seed_target_seconds=seed_generator.target_seconds,
            expander_target_seconds=runtime_window.target_seconds,
        ),
        target_count=target_count,
        candidates_per_request=candidates_per_request,
        max_rounds_per_seed=max_rounds_per_seed,
        request_timeout_seconds=request_timeout_seconds,
        sample_value_limit=sample_value_limit,
        planner_max_paths=planner_max_paths,
        live_seeds_per_turn=live_seeds_per_turn,
        live_max_seeds_per_path=live_max_seeds_per_path,
        live_generator_attempts_per_turn=live_generator_attempts_per_turn,
        live_max_no_gain_turns=live_max_no_gain_turns,
        validate_runs=validate_runs,
        max_runtime_cv=max_runtime_cv,
        max_runtime_spread=max_runtime_spread,
        runtime_window=runtime_window,
    )

    settings.summary_file.parent.mkdir(parents=True, exist_ok=True)
    settings.output_file.parent.mkdir(parents=True, exist_ok=True)
    settings.records_file.parent.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    settings.timing_log_file.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    settings.planner_paths_file.parent.mkdir(parents=True, exist_ok=True)
    settings.planner_constraints_file.parent.mkdir(parents=True, exist_ok=True)
    return settings
