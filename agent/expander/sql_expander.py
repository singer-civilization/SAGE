import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from agent.expander.config import load_settings
from agent.expander.workflow import run_expander


def main() -> None:

    settings = load_settings()
    result = run_expander(settings)
    print(f"accepted_count={result['accepted_count']}")
    print(f"seed_count={result['seed_count']}")
    print(f"summary_file={result['summary_file']}")
    print(f"output_file={result['output_file']}")
    print(f"records_file={result['records_file']}")
    print(f"log_file={result['log_file']}")
    print(f"timing_log_file={result['timing_log_file']}")


if __name__ == "__main__":
    main()
