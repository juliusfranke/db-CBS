import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import yaml
from concurrent.futures import ProcessPoolExecutor
from functools import partial


DATASET_PATH = Path("../results/dataset/")


def try_func(func: callable, *args):
    try:
        return func(*args)
    except:
        if isinstance(args[0], Path):
            raise Exception(f"Error with {str(args[0])}")
        raise Exception(f"Error with {args}")


def _load_instance(instance_data_path: Path) -> List[Dict[str, int | float]]:
    instance_data = {}
    with open(instance_data_path, "r") as file:
        instance_dict = yaml.safe_load(file)
    for key, value in instance_dict["environment"].items():
        if isinstance(value, float):
            instance_data[key] = round(value, ndigits=3)
        elif isinstance(value, int):
            instance_data[key] = round(float(value), ndigits=3)
    instance_data["env_theta_start"] = instance_dict["robots"][0]["start"][2]
    instance_data["env_theta_goal"] = instance_dict["robots"][0]["goal"][2]
    instance_data["instance"] = instance_data_path.stem
    return instance_data


def _load_solution(solution_data_path: Path):
    solution_data = []
    delta_0 = float(solution_data_path.parents[2].name)
    instance = str(solution_data_path.parents[3].name)
    with open(solution_data_path, "r") as file:
        solution_dict = yaml.safe_load(file)
    for mp in solution_dict["motion_primitives"]:
        mp_data = {}
        for i, action in enumerate(np.array(mp["actions"]).flatten()):
            mp_data[f"actions_{i}"] = float(action)
        mp_data[f"theta_0"] = float(mp["start"][2])
        mp_data["cost"] = solution_dict["cost"]
        mp_data["delta_0"] = delta_0
        mp_data["instance"] = instance
        solution_data.append(mp_data)
    return solution_data


def load_instances(max_workers=None):
    instance_data_paths = list(DATASET_PATH.glob("*.yaml"))
    instances_data = []
    error_list = []
    print("Loading instance data")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        load_instance = partial(try_func, _load_instance)
        pbar = tqdm(
            [executor.submit(load_instance, path) for path in instance_data_paths]
        )
        pbar.set_description("Error: 0")
        for execution in pbar:
            exception = execution.exception()
            if exception:
                error_list.append(exception)
                pbar.set_description(f"Error: {len(error_list)}")
                continue
            result = execution.result()
            instances_data.append(result)
    [print(error) for error in error_list]
    return pd.DataFrame(instances_data)


def load_solutions(folder: Path, max_workers=None, debug=False):
    if debug:
        solutions_data_paths = list(folder.glob("**/result_dbcbs.yaml"))[:100]
    else:
        solutions_data_paths = list(folder.glob("**/result_dbcbs.yaml"))

    solutions_data = []
    error_list = []
    print("Loading solution data")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        load_solution = partial(try_func, _load_solution)
        pbar = tqdm(
            [executor.submit(load_solution, path) for path in solutions_data_paths]
        )
        pbar.set_description("Error: 0")
        for execution in pbar:
            exception = execution.exception()
            if exception:
                error_list.append(exception)
                pbar.set_description(f"Error: {len(error_list)}")
                continue
            result = execution.result()
            solutions_data.extend(result)

    [print(error) for error in error_list]
    solutions_df = pd.DataFrame(solutions_data).value_counts().reset_index()
    return solutions_df


def main():
    instances_data = load_instances()
    solutions_data = load_solutions(DATASET_PATH / "Baseline l5 n1000_")
    dataset = solutions_data.merge(instances_data, on="instance").drop(
        columns="instance"
    )

    out = Path("../output/dataset_test.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out)
    print(f"Saved as {out}")


if __name__ == "__main__":
    main()
