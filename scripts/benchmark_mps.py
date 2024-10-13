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
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import psutil

# import checker
from benchmark_stats import run_benchmark_stats
from benchmark_table import write_table
import paper_tables
from instance import createRandomInstance, loadAllInstances


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
    result["instance"] = task.instance
    return result


def main():
    rand_instance_config = {
        "env_min": [8, 8],
        "env_max": [8, 8],
        "obstacle_min": 0.8,
        "obstacle_max": 0.8,
        "allow_disconnect": False,
        "grid_size": 1,
        "save": True,
        "dataset": True,
    }
    old_random_instances = loadAllInstances()
    new_random_instances = [
        createRandomInstance(**rand_instance_config) for _ in range(0)
    ]
    random_instances = {
        name: info for (name, info) in new_random_instances
    } | old_random_instances
    instances = [
        # "alcove_unicycle_single",
        # "bugtrap_single",
        *[rand_inst_name for rand_inst_name in random_instances.keys()],
        # "parallelpark_single",
    ]
    # breakpoint()

    # instances = [
    #     # "alcove_unicycle_single",
    #     "bugtrap_single",
    #     # "parallelpark_single",
    # ]

    alg = "db-cbs"
    trials = 50
    timelimit = 5
    # test_sizes = [25, 50, 100]
    # test_sizes = [50, 100, 250]
    # test_sizes = [n for n in range(5, 105, 5)]
    # test_sizes = np.arange(10, 110, 10, dtype=int).tolist()
    # test_sizes = [1,2,3,4] + [n for n in range(5, 105, 5)]
    # test_sizes = [50,100]
    test_sizes = [50, 75, 100]
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
                "path": unicycle_path / "unicycle1_v0_n50000_l5.bin",
                "name": "Baseline",
            },
            # "Baseline": [
            #     {
            #         "path": unicycle_path / "unicycle1_v0_n1000_l5.yaml",
            #         "name": "Baseline l5 n1000",
            #     },
            # {
            #     "path": unicycle_path / "unicycle1_v0_n10000_l5.yaml",
            #     "name": "Baseline l5 n10000",
            # },
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
    mps["Baseline"] = []
    models = [
        # {
        #     "instance": "",
        #     "modelName": "test",
        #     "name": "wo condition (probability)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "relp_mse",
        #     "name": "wo condition (probability)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_mse",
        #     "name": "wo condition (location)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_rot2x2_mse",
        #     "name": "wo condition (location, svd)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "relp_envt_mse",
        #     "name": "theta (probability)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_envt_mse",
        #     "name": "theta (location)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "relp_po_mse",
        #     "name": "p_obstacle (probability)",
        #     "length": 5,
        # },
        {
            "instance": "",
            "modelName": "l_po_r2_mse",
            "name": "p_obstacle (location, mse)",
            "length": 5,
        },
        {
            "instance": "",
            "modelName": "l_po_r2_mae",
            "name": "p_obstacle (location, mae)",
            "length": 5,
        },
        {
            "instance": "",
            "modelName": "l_po_r2_sh",
            "name": "p_obstacle (location, sh)",
            "length": 5,
        },
        # {
        #     "instance": "",
        #     "modelName": "l_po_rot2x2_mse",
        #     "name": "p_obstacle (location, svd out)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_po_rot2x2_in_mse",
        #     "name": "p_obstacle (location, svd)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_po_mse_rot",
        #     "name": "p_obstacle (location, rot)",
        #     "length": 5,
        # },
        # {
        #     "instance": "",
        #     "modelName": "l_envt_po_mse",
        #     "name": "theta + p_obstacle (location)",
        #     "length": 5,
        # },
    ]
    # models = []
    sample_tasks = []
    for instance in instances:
        for delta in delta_0s:
            for model in models:
                path = (
                    unicycle_path
                    / "diff"
                    / instance
                    / model["name"]
                    / str(delta)
                    / diffusion_name.format("MODEL_SIZE", str(model["length"]), "TRIAL")
                )
                if len(
                    list(
                        path.parent.glob(f"model_unicycle_n*_l{model['length']}_*.yaml")
                    )
                ) >= trials * len(model_sizes):
                    continue

                sample_tasks.append(
                    [
                        "python3",
                        "../master_thesis_code/main.py",
                        "export",
                        model["modelName"],
                        "-d",
                        str(delta),
                        "-r",
                        str(trials),
                        "-s",
                        *[str(model_size) for model_size in model_sizes],
                        "-o",
                        str(path),
                        "-i",
                        f"../example/{instance}.yaml",
                    ]
                )

    error_list = []
    breakpoint()
    with mp.Pool(7) as p:
        for _ in tqdm(
            p.imap_unordered(
                subprocess.run,
                # partial(
                #     subprocess.run, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                # ),
                sample_tasks,
            ),
            total=len(sample_tasks),
        ):
            pass
        # pbar = tqdm(total=len(sample_tasks))
        # pbar.set_description(f"Error: {len(error_list)}")
        # for execution in [
        #     executor.submit(
        #         subprocess.run(
        #             sample_task, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        #         )
        #     )
        #     for sample_task in sample_tasks
        # ]:
        #     exception = execution.exception()
        #     if exception:
        #         error_list.append(exception)
        #         pbar.set_description(f"Error: {len(error_list)}")
        #         pbar.update(1)
        #         continue
        #     pbar.update(1)
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
        "instance": [],
    }
    # result = execute_task(tasks[0])
    # breakpoint()
    error_list = []
    with ProcessPoolExecutor() as executor:
        pbar = tqdm(total=len(tasks))
        pbar.set_description(f"Error: {len(error_list)}")
        for execution in [executor.submit(execute_task, task) for task in tasks]:
            exception = execution.exception()
            if exception:
                error_list.append(exception)
                pbar.set_description(f"Error: {len(error_list)}")
                pbar.update(1)
                continue
            result = execution.result()
            for key, value in result.items():
                results[key].append(value)
            pbar.update(1)
    pbar.close()
    # parallel = True
    # if parallel and len(tasks) > 1:
    #     use_cpus = psutil.cpu_count(logical=False) - 1
    #     print("Using {} CPUs".format(use_cpus))
    #     with mp.Pool(use_cpus) as p:
    #         for result in tqdm.tqdm(
    #             p.imap_unordered(execute_task, tasks), total=len(tasks)
    #         ):
    #             for key, value in result.items():
    #                 results[key].append(value)
    # else:
    #     for task in tasks:
    #         result = execute_task(task)
    #         for key, value in result.items():
    #             results[key].append(value)

    #         results["mp_name"].append(task.mp_name)
    #         results["size"].append(task.size)

    # results_mean.sort_values("idx")

    # order = ["Baseline", *[diff["name"] for diff in mps["Diffusion"].values()]]
    # sns.boxplot(results, x="size", y="success", hue="mp_name", hue_order=order, medianprops={"color": "r", "linewidth":3})
    # fig_scs = plt.figure(figsize=(16,9))
    # fig_dur = plt.figure(figsize=(16,9))
    # fig_cost = plt.figure(figsize=(16,9))
    # ax_scs = fig_scs.axes
    # ax_dur = fig_dur.axes
    # ax_cost = fig_cost.axes
    # if len(delta_0s) != 1:
    #     style = "delta_0"
    # else:
    #     style = None
    # fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    # sns.lineplot(results, x="size", y="success", hue="mp_name", style=style, ax=ax)
    # fig.savefig("../results/plot_success.png")
    # fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    # sns.lineplot(
    #     results,
    #     x="size",
    #     y="duration_dbcbs",
    #     hue="mp_name",
    #     style=style,
    #     ax=ax,
    #     legend=True,
    # )
    # fig.savefig("../results/plot_duration.png")
    # fig, ax = plt.subplots(1, sharex=True, figsize=(16, 9))
    # sns.lineplot(
    #     results,
    #     x="size",
    #     y="cost",
    #     hue="mp_name",
    #     style=style,
    #     ax=ax,
    #     legend=True,
    # )
    # fig.savefig("../results/plot_cost.png")
    results = pd.DataFrame(results)

    instances = pd.DataFrame(
        [info | {"instance": name} for name, info in random_instances.items()]
    )
    breakpoint()
    results = results.merge(instances, on="instance", how="left")
    # def categorize_string(s):
    #     if "location" in s:
    #         return "location"
    #     elif "probability" in s:
    #         return "probability"
    #     else:
    #         return "Baseline"

    # def remove_cat(s):
    #     s = s.replace(" (probability)", "").replace(" (location)", "")
    #     return s

    # results["cat"] = results["mp_name"].apply(categorize_string)
    # results["mp_name"] = results["mp_name"].apply(remove_cat)

    results.to_parquet("../results/bench.parquet")

    # handles, labels = ax[0].get_legend_handles_labels()
    # ax[0].get_legend().remove()
    # fig.legend(handles, labels, loc="lower center", ncol=4)

    # fig_scs.savefig("../results/success.png")
    # fig_dur.savefig("../results/duration.png")
    # fig_cost.savefig("../results/cost.png")


if __name__ == "__main__":
    main()
