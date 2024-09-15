from functools import partial
import fnmatch
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
from instance import createRandomInstance


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
    delta_0: float


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


def execute_task(task: ExecutionTask) -> Dict[str, str | float | None]:
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
        / str(task.delta_0)
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
        mycfg["delta_0"] = task.delta_0
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
    rand_instance_config = {
        "env_min": [4, 4],
        "env_max": [8, 8],
        "obstacle_min": 0.1,
        "obstacle_max": 0.5,
        "allow_disconnect": False,
        "grid_size": 1,
        "save": True,
    }
    random_instances = [createRandomInstance(**rand_instance_config) for _ in range(5)]
    instances = [
        # "alcove_unicycle_single",
        # "bugtrap_single",
        *[rand_inst.name for rand_inst in random_instances],
        # "parallelpark_single",
    ]
    # instances = [
    #     # "alcove_unicycle_single",
    #     "bugtrap_single",
    #     # "parallelpark_single",
    # ]

    alg = "db-cbs"
    trials = 5
    timelimit = 10
    # test_sizes = [25, 50, 100]
    # test_sizes = [50, 100, 250]
    test_sizes = [n for n in range(5, 105, 5)]
    test_sizes = np.arange(10, 110, 10, dtype=int).tolist()
    # test_sizes = [1,2,3,4] + [n for n in range(5, 105, 5)]
    # test_sizes= [100]
    # test_sizes = [n for n in range(50, 60, 5)]
    # delta_0s = [0.3, 0.5, 0.7]
    delta_0s = [0.5]
    # breakpoint()
    unicycle_path = Path("../new_format_motions/unicycle1_v0")
    diffusion_name = "model_unicycle_n{}_l{}_{}.yaml"
    model_sizes = test_sizes
    # model_sizes = []
    mps = {
        "Baseline": [
            {
                "path": unicycle_path / "unicycle1_v0_n1000_l5.yaml",
                "name": "Baseline l5 n1000",
            },
            {
                "path": unicycle_path / "unicycle1_v0_n10000_l5.yaml",
                "name": "Baseline l5 n10000",
            },
            # {
            #     "path": unicycle_path / "unicycle1_v0_n1000_l10.yaml",
            #     "name": "Baseline l10",
            # },
        ],
        "Diffusion": [],
        # n: {
        #     "path": unicycle_path / diffusion_name.format(str(n)),
        #     "name": f"n = {n}",
        # }
        # for n in model_sizes
    }
    # mps["Baseline"] = []
    models = [
        # {
        #     "instance": "parallelpark_single",
        #     "modelName": "parallelpark_l5",
        #     "path": "../../master_thesis_code/bugtrap_l5.pt",
        #     "name": "Model l5",
        #     "length": 5,
        # },
        # {
        #     "instance": "bugtrap_single",
        #     "modelName": "bugtrap_l5_delta",
        #     "path": "../../master_thesis_code/bugtrap_l5_delta.pt",
        #     "name": "Model l5 delta",
        #     "length": 5,
        # },
        # {
        #     "instance": "bugtrap_single",
        #     "modelName": "bugtrap_l5",
        #     "path": "../../master_thesis_code/bugtrap_l5.pt",
        #     "name": "Model l5",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "rand_env_l5",
        #     "path": "../../master_thesis_code/bugtrap_l5_new.pt",
        #     "name": "Model l5 env",
        #     "length": 5,
        # },
        # {
        #     "modelName": "bugtrap_l10",
        #     "path": "../../master_thesis_code/bugtrap_l10.pt",
        #     "name": "Model l10",
        #     "length": 10,
        # },
    ]
    # models = []
    sample_tasks = []
    for instance in instances:
        for trial in range(trials):
            for model_size in model_sizes:
                for delta in delta_0s:
                    for model in models:
                        path = (
                            unicycle_path
                            / "diff"
                            / instance
                            / model["name"]
                            / str(delta)
                            / diffusion_name.format(
                                str(model_size), str(model["length"]), str(trial)
                            )
                        )
                        if path.exists():
                            continue

                        sample_tasks.append(
                            [
                                "python3",
                                "../master_thesis_code/main.py",
                                "export",
                                model["modelName"],
                                "-d",
                                str(delta),
                                "-s",
                                str(model_size),
                                "-o",
                                str(path),
                                "-i",
                                f"../example/{instance}.yaml",                   
                            ]
                        )
    # breakpoint()

    use_cpus = psutil.cpu_count(logical=False) - 1
    print("Using {} CPUs".format(use_cpus))
    print("Generating datasets")
    with mp.Pool(use_cpus) as p:
        for _ in tqdm.tqdm(
            p.imap_unordered(
                # subprocess.run,
                partial(
                    subprocess.run, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ),
                sample_tasks,
            ),
            total=len(sample_tasks),
        ):
            pass
    print("done generating datasets")
    # subprocess.run(
    #     sample_task, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    #     )

    # breakpoint()
    tasks = []
    for instance in instances:
        for trial in range(trials):
            for delta_0 in delta_0s:
                for size in test_sizes:
                    for baseline in mps["Baseline"]:
                        tasks.append(
                            ExecutionTask(
                                instance=instance,
                                alg=alg,
                                trial=trial,
                                timelimit=timelimit,
                                size=size,
                                mp_path=str(baseline["path"]),
                                mp_name=baseline["name"],
                                delta_0=delta_0,
                            )
                        )
                    for model_size in model_sizes:
                        if not model_size == size:
                            continue
                        for model in models:
                            path = (
                                unicycle_path
                                / "diff"
                                / instance 
                                / model["name"]
                                / str(delta_0)
                                / diffusion_name.format(
                                    str(model_size), str(model["length"]), str(trial)
                                )
                            )
                            tasks.append(
                                ExecutionTask(
                                    instance=instance,
                                    alg=alg,
                                    trial=trial,
                                    timelimit=timelimit,
                                    size=size,
                                    mp_path=str(path),
                                    mp_name=model["name"],
                                    delta_0=delta_0,
                                )
                            )

    # breakpoint()
    results = {
        "mp_name": [],
        "size": [],
        "success": [],
        "cost": [],
        "duration_dbcbs": [],
        "delta_0": [],
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

    # results_mean.sort_values("idx")

    # order = ["Baseline", *[diff["name"] for diff in mps["Diffusion"].values()]]
    # sns.boxplot(results, x="size", y="success", hue="mp_name", hue_order=order, medianprops={"color": "r", "linewidth":3})
    # fig_scs = plt.figure(figsize=(16,9))
    # fig_dur = plt.figure(figsize=(16,9))
    # fig_cost = plt.figure(figsize=(16,9))
    # ax_scs = fig_scs.axes
    # ax_dur = fig_dur.axes
    # ax_cost = fig_cost.axes
    if len(delta_0s) != 1:
        style = "delta_0"
    else:
        style = None
    fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    sns.lineplot(results, x="size", y="success", hue="mp_name", style=style, ax=ax)
    fig.savefig("../results/plot_success.png")
    fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    sns.lineplot(
        results,
        x="size",
        y="duration_dbcbs",
        hue="mp_name",
        style=style,
        ax=ax,
        legend=True,
    )
    fig.savefig("../results/plot_duration.png")
    fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    sns.lineplot(
        results,
        x="size",
        y="cost",
        hue="mp_name",
        style=style,
        ax=ax,
        legend=True,
    )
    fig.savefig("../results/plot_cost.png")
    results = pd.DataFrame(results)
    results.to_csv("../results/test.csv")

    # handles, labels = ax[0].get_legend_handles_labels()
    # ax[0].get_legend().remove()
    # fig.legend(handles, labels, loc="lower center", ncol=4)

    # fig_scs.savefig("../results/success.png")
    # fig_dur.savefig("../results/duration.png")
    # fig_cost.savefig("../results/cost.png")


if __name__ == "__main__":
    main()
