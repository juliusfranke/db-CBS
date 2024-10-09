import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import yaml
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import tempfile


DATASET_PATH = Path("../results/dataset/")


def try_func(func: callable, *args):
    try:
        return func(*args)
    except:
        if isinstance(args[0], Path):
            raise Exception(f"Error with {str(args[0])}")
        raise Exception(f"Error with {args}")


def _load_instance(instance_data_path: Path):
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
    total_mp = len(solution_dict["motion_primitives"]) - 1
    for i, mp in enumerate(solution_dict["motion_primitives"]):
        mp_data = {}
        for j, action in enumerate(np.array(mp["actions"]).flatten()):
            mp_data[f"actions_{j}"] = float(action)
        mp_data[f"theta_0"] = float(mp["start"][2])
        mp_data["cost"] = solution_dict["cost"]
        mp_data["delta_0"] = delta_0
        mp_data["instance"] = instance
        mp_data["location"] = i / total_mp
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


def load_solutions_partial(
    paths: List, pbar: tqdm, error_list: List, temp_file, max_workers=None
):
    solutions_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        load_solution = partial(try_func, _load_solution)
        executions = [executor.submit(load_solution, path) for path in paths]
        for execution in executions:
            exception = execution.exception()
            pbar.update()
            if exception:
                error_list.append(exception)
                pbar.set_description(f"Error: {len(error_list)}")
                continue
            result = execution.result()
            solutions_data.extend(result)
    # solutions_df = pd.DataFrame(solutions_data).value_counts().reset_index()
    solutions_df = pd.DataFrame(solutions_data)
    solutions_df.to_parquet(temp_file)


def load_solutions(folder: Path, max_workers=None, debug=False):
    solutions_data_paths = list(folder.glob("**/result_dbcbs.yaml"))

    tempdir = tempfile.gettempdir()
    temp_files = []
    error_list = []
    print("Loading solution data")
    pbar = tqdm(total=len(solutions_data_paths))
    pbar.set_description("Error: 0")
    max_simult = 1000
    for i in range(int(np.ceil(len(solutions_data_paths) / max_simult))):
        temp_file = f"{tempdir}/{i}.parquet"
        temp_files.append(temp_file)
        paths = solutions_data_paths[i * max_simult : (i + 1) * max_simult]
        load_solutions_partial(paths, pbar, error_list, temp_file)
        if i == 2 and debug:
            break

    pbar.close()
    [print(error) for error in error_list]
    solutions_df = pd.concat([pd.read_parquet(file) for file in temp_files])
    # for temp_file in temp_files:
    #     Path(temp_file).unlink()
    # solutions_df = pd.DataFrame(solutions_data).value_counts().reset_index()
    # solutions_df = solutions_df.value_counts().reset_index()
    return solutions_df


def main():
    # instances_data = load_instances()
    # solutions_data = _load_solution(
    #     DATASET_PATH
    #     / "test"
    #     / "000aa150-d957-430a-8bc4-978ad90633b2"
    #     / "0.2"
    #     / "100"
    #     / "000"
    #     / "result_dbcbs.yaml"
    # )
    # solutions_data = pd.DataFrame(solutions_data)
    # dataset = solutions_data.merge(instances_data, on="instance").drop(
    #     columns="instance"
    # )
    # breakpoint()
    solutions_data = pd.read_parquet("../results/dataset/data.parquet")
    insts = list(solutions_data["instance"].unique())
    instances_data = load_instances()
    instances_data = instances_data[instances_data["instance"].isin(insts)]
    dataset = solutions_data.merge(instances_data, on="instance")

    # solutions_data = load_solutions(DATASET_PATH / "Baseline l5 n50000", debug=False)
    breakpoint()

    dataset = (
        solutions_data.merge(instances_data, on="instance")
        .drop(columns="instance")
        .value_counts()
        .reset_index()
    )

    out = Path("../output/rand_env_40k.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out)
    print(f"Saved as {out}")


if __name__ == "__main__":
    main()
