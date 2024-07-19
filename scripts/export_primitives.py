import yaml
from pathlib import Path

RESULT_PATH = Path("../results")
OUTPUT_PATH = Path("../output/")
def main():
    for folder in RESULT_PATH.iterdir():
        data = []
        problem_name = folder.name
        solutions_path = folder / "db-cbs"
        if not solutions_path.exists():
            print(f"{problem_name} has no solution from db-cbs")
            continue
        for solution in solutions_path.iterdir():
            output_file = solution / "result_dbcbs.yaml"
            if not output_file.exists():
                print(f"{output_file} does not exist")
                continue
            with open(output_file, "r") as file:
                solution_data = yaml.safe_load(file)
            primitives = solution_data["motion_primitives"]
            data.extend(primitive for primitive in primitives)

        out = OUTPUT_PATH / (problem_name + ".yaml")
        with open(out, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=None)

if __name__ == "__main__":
    main()

