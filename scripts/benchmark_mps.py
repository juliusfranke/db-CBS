from typing import Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from main_ompl import run_ompl
from main_s2m2 import run_s2m2
from main_kcbs import run_kcbs
from main_dbcbs_mod import run_dbcbs
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass
import multiprocessing as mp
import tqdm
import psutil

# import checker
from benchmark_stats import run_benchmark_stats
from benchmark_table import write_table
import paper_tables


@dataclass
class ExecutionTask:
    """Class for keeping track of an item in inventory."""

    # env: Path
    # cfg: Path
    # result_folder: Path
    instance: str
    alg: str
    trial: int
    timelimit: float
    size: int
    mp_path: str
    mp_name: str


def run_visualize(script, filename_env, filename_result):
    subprocess.run(
        [
            "python3",
            script,
            filename_env,
            "--result",
            filename_result,
            "--video",
            filename_result.with_suffix(".mp4"),
        ]
    )


def run_checker(filename_env, filename_result, filename_log):
    with open(filename_log, "w") as f:
        cmd = [
            "./dynoplan/dynobench/check_trajectory_multirobot",
            "--result_file",
            filename_result,
            "--env_file",
            filename_env,
            "--models_base_path",
            "../dynoplan/dynobench/models/",
            "--goal_tol",
            "999999",
        ]
        # print(subprocess.list2cmdline(cmd))
        out = subprocess.run(cmd, stdout=f, stderr=f)
    return out.returncode == 0


def run_search_visualize(script, filename_env, filename_trajs, filename_result):
    subprocess.run(
        [
            "python3",
            script,
            filename_env,
            "--trajs",
            filename_trajs,
            "--result",
            filename_result,
            "--video",
            filename_result.with_suffix(".mp4"),
        ]
    )


def execute_task(task: ExecutionTask) -> Dict[str, float | None]:
    scripts_path = Path("../scripts")
    results_path = Path("../results")
    # tuning_path = Path("../tuning")
    env_path = Path().resolve() / "../example"
    env = (env_path / task.instance).with_suffix(".yaml")
    assert env.is_file()

    cfg = env_path / "algorithms.yaml"  # using single alg.yaml
    assert cfg.is_file()

    with open(cfg) as f:
        cfg = yaml.safe_load(f)

    result_folder = (
        results_path
        / task.mp_name
        / task.instance
        / str(task.size)
        # / task.alg
        / "{:03d}".format(task.trial)
    )
    if result_folder.exists():
        # print("Warning! {} exists already. Deleting...".format(result_folder))
        shutil.rmtree(result_folder)
    result_folder.mkdir(parents=True, exist_ok=False)

    # find cfg
    mycfg = cfg[task.alg]
    mycfg = mycfg["default"]
    # wildcard matching
    import fnmatch

    for k, v in cfg[task.alg].items():
        if fnmatch.fnmatch(Path(task.instance).name, k):
            mycfg = {**mycfg, **v}  # merge two dictionaries

    if Path(task.instance).name in cfg[task.alg]:
        mycfg_instance = cfg[task.alg][Path(task.instance).name]
        mycfg = {**mycfg, **mycfg_instance}  # merge two dictionaries

    # print("Using configurations ", mycfg)

    if task.alg == "db-cbs":
        # breakpoint()
        mycfg["num_primitives_0"] = task.size
        mycfg["mp_path"] = task.mp_path
        mycfg["mp_name"] = task.mp_name
        result = run_dbcbs(str(env), str(result_folder), task.timelimit, mycfg)
        visualize_files = [p.name for p in result_folder.glob("result_*")]
        check_files = [p.name for p in result_folder.glob("result_dbcbs_opt*")]
        search_plot_files = [p.name for p in result_folder.glob("expanded_trajs*")]
    else:
        raise NotImplementedError

    for file in check_files:
        if not run_checker(
            env, result_folder / file, (result_folder / file).with_suffix(".check.txt")
        ):
            print("WARNING: CHECKER FAILED -> DELETING stats!")
            (result_folder / "stats.yaml").unlink(missing_ok=True)

    # vis_script = scripts_path / "visualize.py"
    # for file in visualize_files:
    #     run_visualize(vis_script, env, result_folder / file)

    # search_viz_script = scripts_path / "visualize_search.py"
    # if(len(search_plot_files) > 0):
    #   for file in search_plot_files:
    #       run_search_visualize(search_viz_script, result_folder / file)
    return result


def main():
    instances = [
        # "alcove_unicycle_single",
        "bugtrap_single",
    ]

    alg = "db-cbs"
    trials = 10
    timelimit = 3
    # test_sizes = [25, 50, 100, 250]
    # test_sizes = [50, 100, 250]
    test_sizes = [n for n in range(5, 105, 5)]
    # breakpoint()
    unicycle_path = Path("../new_format_motions/unicycle1_v0")
    diffusion_name = "model_unicycle_bugtrap_n{}_l5_{}.yaml"
    model_sizes = test_sizes
    mps = {
        "Baseline": {
            "path": unicycle_path / "unicycle1_v0_n1000_l5.yaml",
            "name": "Baseline",
        },
        "Diffusion": []
            # n: {
            #     "path": unicycle_path / diffusion_name.format(str(n)),
            #     "name": f"n = {n}",
            # }
            # for n in model_sizes
    }
    for trial in range(trials):
        for model_size in model_sizes:
            path = unicycle_path / diffusion_name.format(str(model_size), str(trial))
            # mps["Diffusion"].append({"path": path, "name": f"n = {model_size}","n":model_size})
            if path.exists():
                continue
            
            subprocess.run(
                [
                    "python3",
                    "../../master_thesis_code/main.py",
                    "export",
                    "../../master_thesis_code/bugtrap_rel_logistic.pt",
                    "-s",
                    str(model_size),
                    "-o",
                    str(path),
                ]
            )

    breakpoint()
    tasks = []
    for instance in instances:
        for trial in range(trials):
            for size in test_sizes:
                tasks.append(
                    ExecutionTask(
                        instance,
                        alg,
                        trial,
                        timelimit,
                        size,
                        str(mps["Baseline"]["path"]),
                        mps["Baseline"]["name"],
                    )
                )
                for model_size in model_sizes:
                    if not model_size == size:
                        continue
                    path = unicycle_path / diffusion_name.format(str(model_size), str(trial))
                    tasks.append(
                        ExecutionTask(
                            instance,
                            alg,
                            trial,
                            timelimit,
                            size,
                            str(path),
                            # model["name"],
                            "model",
                        )
                    )

    # breakpoint()
    results = {
        "mp_name": [],
        "size": [],
        "success": [],
        "cost": [],
        "duration_dbcbs": [],
    }
    parallel = True
    if parallel and len(tasks) > 1:
        use_cpus = psutil.cpu_count(logical=False) - 1
        print("Using {} CPUs".format(use_cpus))
        with mp.Pool(use_cpus) as p:
            for result in tqdm.tqdm(
                p.imap_unordered(execute_task, tasks), total=len(tasks)
            ):
                for key, value in result.items():
                    results[key].append(value)
    else:
        for task in tasks:
            result = execute_task(task)
            for key, value in result.items():
                results[key].append(value)

            results["mp_name"].append(task.mp_name)
            results["size"].append(task.size)

    breakpoint()
    results_mean = pd.DataFrame(
        pd.DataFrame(results).groupby(["mp_name", "size"]).mean()
    )
    # results_mean.sort_values("idx")

    # order = ["Baseline", *[diff["name"] for diff in mps["Diffusion"].values()]]
    # sns.boxplot(results, x="size", y="success", hue="mp_name", hue_order=order, medianprops={"color": "r", "linewidth":3})
    sns.lineplot(results, x="size", y="success", hue="mp_name")
    plt.savefig("../results/success.png")
    plt.clf()
    sns.lineplot(results, x="size", y="duration_dbcbs", hue="mp_name")
    # sns.boxplot(results, x="size", y="duration_dbcbs", hue="mp_name", hue_order=order, medianprops={"color": "r","linewidth":3})
    # sns.barplot(results_mean, x="size", y="duration_dbcbs", hue="mp_name")
    plt.savefig("../results/duration.png")
    plt.clf()
    # sns.boxplot(results, x="size", y="cost", hue="mp_name", hue_order=order, medianprops={"color": "r","linewidth":3})
    sns.lineplot(results, x="size", y="cost", hue="mp_name")

    plt.savefig("../results/cost.png")

    plt.clf()
    # sns.barplot(results_mean, x="size", y="cost", hue="mp_name")


if __name__ == "__main__":
    main()
