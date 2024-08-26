from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from shapely import Point, STRtree, Polygon
from tqdm import tqdm
import geopandas as gpd

RESULT_PATH = Path("../results/dataset/")
OUTPUT_PATH = Path("../output/")


class MotionPrimitive:
    def __init__(
        self,
        start: np.ndarray,
        actions: np.ndarray,
        delta_0: float,
        goal: np.ndarray | None = None,
        states: np.ndarray | None = None,
    ) -> None:
        self.rel_probability: float = 0.0
        self.delta_0: float = delta_0
        self.cdf: float = 0.0
        self.count: int = 1
        self.start = np.array([0, 0, start[2]])
        self.actions = actions
        if goal:
            self.goal = diff(start, goal)
        else:
            self.goal = None
        if states:
            self.states = np.array([diff(start, state) for state in states])
        else:
            self.states = None

    def computeStates(self) -> None:
        if isinstance(self.states, np.ndarray):
            return None
        self.states = calc_unicycle_states(self.actions, self.start)
        self.goal = self.states[-1]

    def toDict(self) -> Dict[str, List[float]]:
        data = {
            key: value.tolist()
            for key, value in self.__dict__.items()
            if (value is not None and isinstance(value, np.ndarray))
        }
        data["count"] = self.count
        data["delta_0"] = self.delta_0
        # data["cdf"] = self.cdf
        data["rel_probability"] = self.rel_probability
        return data

    def __hash__(self) -> int:
        return hash(
            (str(round(self.start[2], 6)), str(self.actions), str(self.delta_0))
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MotionPrimitive):
            return NotImplemented
        if np.allclose(self.start, other.start, atol=1e-06) and np.allclose(
            self.actions, other.actions
        ):
            return True
        return False

    def __repr__(self) -> str:
        if self.goal:
            return f"{self.start} -> {self.goal}"
        return f"{self.start} -> [?,?,?]"


def calc_unicycle_states(actions: np.ndarray, start: np.ndarray, dt: float = 0.1):
    x, y, theta = start
    states = [list(start)]
    for s, phi in actions:
        dx = dt * s * np.cos(theta)
        dy = dt * s * np.sin(theta)
        dtheta = dt * phi

        x += dx
        y += dy
        theta += dtheta
        states.append([x, y, theta])
    return np.array(states)


def diff(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.array([*goal[:2] - start[:2], goal[2]])


def main():
    for model in RESULT_PATH.iterdir():
        model_name = model.name
        print(f"Export data from {model_name} ? (y/n)")
        if input() in ["n", "no"]:
            continue
        for problem in model.iterdir():
            problem_name = problem.name
            params = instance_params(problem_name)
            breakpoint()
            export(model_name, problem_name)


def rect_from_2p(
    p0: List[float | int], p1: List[float | int], mode="center_size"
) -> Polygon:
    if mode == "min_max":
        rect = [(p0[0], p0[1]), (p1[0], p0[1]), (p1[0], p1[1]), (p0[0], p1[1])]
    elif mode == "center_size":
        rect = [
            (p0[0] - p1[0] / 2, p0[1] - p1[1] / 2),
            (p0[0] + p1[0] / 2, p0[1] - p1[1] / 2),
            (p0[0] + p1[0] / 2, p0[1] + p1[1] / 2),
            (p0[0] - p1[0] / 2, p0[1] + p1[1] / 2),
        ]
    else:
        raise NotImplementedError(mode)

    return Polygon(rect)


def instance_params(instance) -> Dict[str, float | int]:
    instance_path = Path("../example") / instance
    instance_path = instance_path.with_suffix(".yaml")
    with open(instance_path, "r") as file:
        content = yaml.safe_load(file)
    env_min = content["environment"]["min"]
    env_max = content["environment"]["max"]
    env = rect_from_2p(env_min, env_max, mode="min_max")
    obstacle_area = []
    obstacles = []
    diff = env
    for obst in content["environment"]["obstacles"]:
        _type = obst["type"]
        center = obst["center"]
        size = obst["size"]
        if _type != "box":
            raise NotImplementedError("Obstacle type not implemented")
        obstacle = rect_from_2p(center, size)
        # if env.overlaps(obstacle):
        #     obstacle = env.intersection(obstacle)
        obstacle_area.append(obstacle.area)
        obstacles.append(obstacle)
        # breakpoint()
        diff = diff.difference(obstacle)
        # p = gpd.GeoSeries(obstacle)
        # p.plot()
    # strtree = STRtree(obstacles)
    gs = gpd.GeoSeries(diff) 
    gs.plot()
    plt.show()
    breakpoint()
    obstacle_area = np.array(obstacle_area)
    n_obstacles = len(obstacle_area)
    mean_size = np.mean(obstacle_area, dtype=float)
    env_width = env_max[0] - env_min[0]
    env_height = env_max[1] - env_min[1]
    density = np.sum(obstacle_area) / env.area
    return {
        "env_width": env_width,
        "env_height": env_height,
        "n_obstacles": n_obstacles,
        "mean_size": mean_size,
        "density": density,
    }


def export(model_name, problem_name):
    debug = True
    data: Dict[str, MotionPrimitive] = {}
    problem_path = RESULT_PATH / model_name / problem_name

    total_len = len([path for path in problem_path.glob("**/stats.yaml")])
    pbar = tqdm(range(total_len))
    for delta in problem_path.iterdir():
        i = 0
        delta_0 = float(delta.name)
        solutions_path = delta / "100"
        for solution in solutions_path.iterdir():
            if debug and i >= 10:
                break
            pbar.update()
            output_file = solution / "result_dbcbs.yaml"
            if not output_file.exists():
                continue
            with open(output_file, "r") as file:
                solution_data = yaml.safe_load(file)
            primitives = solution_data["motion_primitives"]
            for primitive in primitives:
                start = np.array(primitive["start"])
                actions = np.array(primitive["actions"])
                mp = MotionPrimitive(start=start, actions=actions, delta_0=delta_0)
                # breakpoint()
                # check = any(_mp == mp for _mp in data)
                hash = str(mp.__hash__())
                check = hash in data
                # print(check, mp in data)
                if check:
                    data[hash].count += 1
                else:
                    data[hash] = mp
            i += 1
    pbar.close()
    setData = list(data.values())
    print(len(setData))
    dictData = {}
    dictData["theta0"] = [mp.start[2] for mp in setData]
    dictData["s"] = [np.mean(mp.actions[:, 0]) for mp in setData]
    dictData["phi"] = [np.mean(mp.actions[:, 1]) for mp in setData]
    dictData["count"] = [mp.count for mp in setData]
    dictData["delta_0"] = [mp.delta_0 for mp in setData]

    dfData = pd.DataFrame(dictData)
    total_counts = dfData.groupby("delta_0").sum()["count"].to_dict()
    max_counts = dfData.groupby("delta_0").max()["count"].to_dict()
    dfData["rel_probability"] = [
        count / max_counts[delta_0]
        for count, delta_0 in dfData[["count", "delta_0"]].to_numpy()
    ]
    dfData["probability"] = [
        count / total_counts[delta_0]
        for count, delta_0 in dfData[["count", "delta_0"]].to_numpy()
    ]
    # dfData["probability"] = dfData["count"] / total_count
    dfData = dfData.sort_values("probability")
    dfData["cdf"] = dfData["probability"].cumsum()
    dfData = dfData.sort_index()
    for i, mp in enumerate(setData):
        mp.cdf = float(dfData["cdf"][i])
        mp.rel_probability = float(dfData["rel_probability"][i])
    # dfData = pd.DataFrame(dfData[dfData["count"]>1])

    fig, axs = plt.subplots(4, 1)
    # sns.set_style("white")
    # sns.set
    sns.histplot(
        data=dfData,
        x="theta0",
        weights="probability",
        hue="delta_0",
        kde=True,
        bins=50,
        ax=axs[0],
    )
    sns.histplot(
        data=dfData,
        x="s",
        weights="probability",
        hue="delta_0",
        kde=True,
        bins=50,
        ax=axs[1],
    )
    sns.histplot(
        data=dfData,
        x="phi",
        weights="probability",
        hue="delta_0",
        kde=True,
        bins=50,
        ax=axs[2],
    )
    # # sns.lineplot(data=dfData, x="rel_probability", y="count", ax=axs[3])
    # sns.histplot(data=dfData, x="count", ax=axs[3])
    sns.histplot(
        data=dfData,
        x="rel_probability",
        kde=True,
        weights="probability",
        hue="delta_0",
        bins=20,
        ax=axs[3],
    )
    sns.set_palette(sns.color_palette("rocket"))
    plt.show()

    breakpoint()
    dataYaml = [mp.toDict() for mp in setData]

    out = OUTPUT_PATH / (problem_name + f"_l{len(setData[0].actions)}" + ".yaml")
    with open(out, "w") as file:
        yaml.dump(dataYaml, file, default_flow_style=None)


if __name__ == "__main__":
    main()
