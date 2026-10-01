import json
import os
from datetime import datetime
from pathlib import Path


def reset_run_files(*paths: Path) -> None:

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def append_text(path: Path, text: str) -> None:

    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(text)


def log_line(path: Path, message: str) -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    append_text(path, line)
    if os.getenv("EXPANDER_ECHO_LOGS", "").lower() in {"1", "true", "yes", "on"}:
        print(line, end="", flush=True)


def append_sql(path: Path, sql: str) -> None:

    append_text(path, sql.rstrip() + "\n")


def append_jsonl(path: Path, payload: dict) -> None:

    append_text(path, json.dumps(payload, ensure_ascii=False) + "\n")
