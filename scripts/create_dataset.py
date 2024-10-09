import fnmatch
import multiprocessing as mp
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import yaml
import numpy as np
from memory_profiler import profile

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
    # "env_min": [8, 8],
    # "env_max": [16, 16],
    "allow_disconnect": False,
    "grid_size": 1,
    "save": True,
    "dataset": True,
}


# @profile
def main():
    # n_instances = 100
    random_instances = {}

    # a = createRandomInstance(**rand_instance_config, obstacle_min=o, obstacle_max=o)
    # fig, ax = plt.subplots(1)
    # a.plotInstance(ax=ax)
    # plt.show()
    #
    # breakpoint()
    n_repeat = 100
    obstacles = np.arange(0.4, 0.8, 0.05)
    # sizes = np.arange(10, 13, 1)
    sizes = [6, 7, 8, 9, 10]
    # sizes = [10]
    obstacles, sizes_x, sizes_y = np.meshgrid(obstacles, sizes, sizes)
    obstacles = np.repeat(obstacles, n_repeat)
    sizes_x = np.repeat(sizes_x, n_repeat)
    sizes_y = np.repeat(sizes_y, n_repeat)

    print(len(obstacles))
    breakpoint()
    # for o, x, y in tqdm(zip(obstacles, sizes_x, sizes_y), total=len(obstacles)):
    #     inst = createRandomInstance(
    #         **rand_instance_config,
    #         env_min=[x, y],
    #         env_max=[x, y],
    #         obstacle_min=o,
    #         obstacle_max=o,
    #     )
    #     random_instances[inst.name] = inst
    # return None
    # error_list = []
    with ProcessPoolExecutor() as executor:
        pbar = tqdm(total=len(obstacles))
        for execution in [
            executor.submit(
                partial(createRandomInstance, **rand_instance_config),
                env_min=[x, y],
                env_max=[x, y],
                obstacle_min=o,
                obstacle_max=o,
            )
            for o, x, y in zip(obstacles, sizes_x, sizes_y)
        ]:
            # exception = execution.exception()
            # if exception:
            #     error_list.append(exception)
            #     pbar.set_description(f"Error: {len(error_list)}")
            #     pbar.update(1)
            #     continue
            name, info = execution.result()
            random_instances[name] = info
            pbar.update(1)
    pbar.close()

    # for i in range(5):
    #     random_instance = createRandomInstance(**rand_instance_config, obstacle_min=0, obstacle_max=0.1)
    #     random_instances[random_instance.name] = random_instance

    instances = [
        # "alcove_unicycle_single",
        # "bugtrap_single",
        *[rand_inst for rand_inst in random_instances.keys()],
        # "parallelpark_single",
    ]

    alg = "db-cbs"
    trials = 20
    timelimit = 3
    test_size = 100
    # delta_0s = [0.3, 0.4, 0.5, 0.6, 0.7]
    delta_0s = [0.5]

    unicycle_path = Path("../new_format_motions/unicycle1_v0")
    mps = {
        "Baseline": [
            {
                "path": unicycle_path / "unicycle1_v0_n50000_l5.bin",
                "name": "Baseline l5 n50000",
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

    # results = {
    #     "instance": [],
    #     "mp_name": [],
    #     "size": [],
    #     "success": [],
    #     "cost": [],
    #     "duration_dbcbs": [],
    #     "delta_0": [],
    #     "p_obstacles": [],
    #     "area": [],
    # }
    # breakpoint()
    results = {}
    error_list = []
    with ProcessPoolExecutor() as executor:
        pbar = tqdm(total=len(tasks))
        pbar.set_description("(Success/Failure): 0/0")
        success = 0
        failure = 0
        for execution in [executor.submit(execute_task, task) for task in tasks]:
            exception = execution.exception()
            if exception:
                error_list.append(exception)
                # pbar.set_description(f"Error: {len(error_list)}")
                pbar.update(1)
                continue
            result = execution.result()
            # pbar.write(result["success"])
            if result["success"]:
                success += 1
                pbar.set_description(f"(Success/Failure): {success}/{failure}")
            else:
                failure += 1

            results[result["instance"]] = result
            # for key, value in result.items():
            #     results[key].append(value)
            # if result["instance"] in random_instances.keys():
            #     results["p_obstacles"].append(
            #         random_instances[result["instance"]].env.info["p_obstacles"]
            #     )
            #     results["area"].append(
            #         random_instances[result["instance"]].env.info["area"]
            #     )
            # else:
            #     results["p_obstacles"].append(None)
            pbar.update(1)
    pbar.close()

    results = pd.DataFrame(results).transpose()
    # instances = pd.DataFrame(
    #     [r.env.info | {"instance": name} for name, r in random_instances.items()]
    # )
    instances = pd.DataFrame(
        [info | {"instance": name} for name, info in random_instances.items()]
    )
    results = results.merge(instances, on="instance")
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
            results,
            x="delta_0",
            y="duration_dbcbs",
            hue="instance",
            ax=ax[1],
            legend=False,
        )
        sns.lineplot(
            results, x="delta_0", y="cost", hue="instance", ax=ax[2], legend=False
        )

        plt.setp(ax, xticks=delta_0s)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].get_legend().remove()
        fig.legend(handles, labels, loc="lower center", ncol=5)

        fig.savefig("../results/creation_delta.png")
    else:
        fig, ax = plt.subplots(3, figsize=(16, 9), sharex=True)
        sns.lineplot(results, x="p_obstacles", y="success", hue="area", ax=ax[0])
        sns.lineplot(results, x="p_obstacles", y="duration_dbcbs", hue="area", ax=ax[1])
        sns.lineplot(results, x="p_obstacles", y="cost", hue="area", ax=ax[2])
        fig.savefig("../results/creation.png")

    # fig_scs.savefig("../results/success.png")
    # fig_dur.savefig("../results/duration.png")
    # fig_cost.savefig("../results/cost.png")
    print("plotting done")
    # for random_instance in random_instances.values():
    #     dataset_instance = Path("../results") / "dataset" / random_instance.name
    #     dataset_instance = dataset_instance.with_suffix(".yaml")
    #     random_instance.save(dataset_instance, extended=True)
    print("complete")


if __name__ == "__main__":
    main()
