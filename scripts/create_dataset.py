import fnmatch
import multiprocessing as mp
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import psutil
import seaborn as sns
import tqdm
import pandas as pd
import yaml

# import checker
from main_dbcbs_mod import run_dbcbs
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


def execute_task(task: ExecutionTask) -> Dict[str, float | str | None]:
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
        / "dataset"
        / task.mp_name
        / task.instance
        / str(task.delta_0)
        / str(task.size)
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
    result["instance"] = task.instance
    return result


rand_instance_config = {
    "env_min": [4, 4],
    "env_max": [8, 8],
    "obstacle_min": 0.1,
    "obstacle_max": 0.5,
    "allow_disconnect": False,
    "grid_size": 1,
    "save": True,
}


def main():
    random_instances = [createRandomInstance(**rand_instance_config) for _ in range(100)]
    instances = [
        # "alcove_unicycle_single",
        # "bugtrap_single",
        *[rand_inst.name for rand_inst in random_instances],
        # "parallelpark_single",
    ]

    alg = "db-cbs"
    trials = 50
    timelimit = 5
    test_size = 500
    # delta_0s = [0.3, 0.4, 0.5, 0.6, 0.7]
    delta_0s = [0.5]

    unicycle_path = Path("../new_format_motions/unicycle1_v0")
    mps = {
        "Baseline": [
            {
                "path": unicycle_path / "unicycle1_v0_n10000_l5.yaml",
                "name": "Baseline l5 n10000",
            },
        ]
    }
    tasks = []
    for instance in instances:
        for trial in range(trials):
            for delta_0 in delta_0s:
                for baseline in mps["Baseline"]:
                    tasks.append(
                        ExecutionTask(
                            instance=instance,
                            alg=alg,
                            trial=trial,
                            timelimit=timelimit,
                            size=test_size,
                            mp_path=str(baseline["path"]),
                            mp_name=baseline["name"],
                            delta_0=delta_0,
                        )
                    )

    results = {
        "instance": [],
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
    # breakpoint()
    if len(delta_0s) > 1:
        fig, ax = plt.subplots(3, sharex=True, figsize=(16, 9))
        sns.lineplot(results, x="delta_0", y="success", hue="instance", ax=ax[0])
        sns.lineplot(
            results, x="delta_0", y="duration_dbcbs", hue="instance", ax=ax[1], legend=False
        )
        sns.lineplot(results, x="delta_0", y="cost", hue="instance", ax=ax[2], legend=False)

        plt.setp(ax, xticks=delta_0s)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].get_legend().remove()
        fig.legend(handles, labels, loc="lower center", ncol=5)

        fig.savefig("../results/creation_delta.png")
    else:
        fig, ax = plt.subplots(3, figsize=(16, 9))
        sns.boxplot(results, x="success", ax=ax[0])
        sns.boxplot(results, x="duration_dbcbs", ax=ax[1])
        sns.boxplot(results, x="cost", ax=ax[2])
        fig.savefig("../results/creation.png")

    # fig_scs.savefig("../results/success.png")
    # fig_dur.savefig("../results/duration.png")
    # fig_cost.savefig("../results/cost.png")
    for random_instance in random_instances:
        dataset_instance = Path("../results") / "dataset" / random_instance.name  
        dataset_instance = dataset_instance.with_suffix(".yaml")
        random_instance.save(dataset_instance, extended = True)


if __name__ == "__main__":
    main()
