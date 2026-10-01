import os

from agent.expander.live_generate_expand import run_live_workflow


def main() -> None:

    os.environ.setdefault("EXPANDER_ECHO_LOGS", "1")
    result = run_live_workflow(emit_seed_output=True)
    print(f"generated_seed_count={result['generated_seed_count']}")
    print(f"accepted_count={result['accepted_count']}")
    print(f"used_path_count={result['used_path_count']}")
    print(f"seed_file={result['seed_file']}")
    print(f"output_file={result['output_file']}")
    print(f"records_file={result['records_file']}")
    print(f"log_file={result['log_file']}")
    print(f"timing_log_file={result['timing_log_file']}")
    print(f"planner_paths_file={result['planner_paths_file']}")
    print(f"planner_constraints_file={result['planner_constraints_file']}")


if __name__ == "__main__":
    main()
