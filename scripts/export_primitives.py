from functools import total_ordering
from typing import Dict, List
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

RESULT_PATH = Path("../results/Randomly generated/")
OUTPUT_PATH = Path("../output/")


class MotionPrimitive:
    def __init__(
        self,
        start: np.ndarray,
        actions: np.ndarray,
        goal: np.ndarray | None = None,
        states: np.ndarray | None = None,
    ) -> None:
        self.rel_probability: float = 0.0
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
        # data["cdf"] = self.cdf
        data["rel_probability"] = self.rel_probability
        return data

    def __hash__(self) -> int:
        return hash((str(round(self.start[2], 6)), str(self.actions)))

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
    # breakpoint()
    for folder in RESULT_PATH.iterdir():
        skipped = 0
        data: Dict[str, MotionPrimitive] = {}
        problem_name = folder.name
        solutions_path = folder / "100"
        if not solutions_path.exists():
            print(f"{problem_name} has no solution from db-cbs")
            continue
        i = 0
        for solution in solutions_path.iterdir():
            # if i > 10:
            #     break
            output_file = solution / "result_dbcbs.yaml"
            if not output_file.exists():
                print(f"{output_file} does not exist")
                continue
            with open(output_file, "r") as file:
                solution_data = yaml.safe_load(file)
            primitives = solution_data["motion_primitives"]
            for primitive in primitives:
                start = np.array(primitive["start"])
                actions = np.array(primitive["actions"])
                mp = MotionPrimitive(start=start, actions=actions)
                # breakpoint()
                # check = any(_mp == mp for _mp in data)
                hash = str(mp.__hash__())
                check = hash in data
                # print(check, mp in data)
                if check:
                    # print("dupe")
                    data[hash].count += 1
                else:
                    data[hash] = mp
            i += 1
        # unique_counts = collections.Counter(e for e in data)
        # d = [
        #     {**x, "count": data.count(x)}
        #     for i, x in enumerate(data)
        #     if x not in data[:i]
        # ]
        # test = pd.DataFrame(d)
        # print(test)
        # print(test[test["count"] > 1])
        # d = list(set(data))
        # breakpoint()
        # setData = list(set(data))
        setData = list(data.values())
        print(len(setData))
        breakpoint()
        dictData = {}
        dictData["theta0"] = [mp.start[2] for mp in setData]
        dictData["s"] = [np.mean(mp.actions[:, 0]) for mp in setData]
        dictData["phi"] = [np.mean(mp.actions[:, 1]) for mp in setData]
        dictData["count"] = [mp.count for mp in setData]

        dfData = pd.DataFrame(dictData)
        total_count = dfData["count"].sum()
        count_max = dfData["count"].max()
        dfData["rel_probability"] = dfData["count"]/ count_max
        dfData["probability"] = dfData["count"] / total_count
        dfData = dfData.sort_values("probability")
        dfData["cdf"] = dfData["probability"].cumsum()
        breakpoint()
        dfData = dfData.sort_index()
        for i, mp in enumerate(setData):
            mp.cdf = float(dfData["cdf"][i])
            mp.rel_probability = float(dfData["rel_probability"][i])

        fig, axs = plt.subplots(4, 1)
        sns.histplot(data=dfData, x="theta0", weights="count", kde=True, ax=axs[0])
        sns.histplot(data=dfData, x="s", weights="count", kde=True, ax=axs[1])
        sns.histplot(data=dfData, x="phi", kde=True, weights="count", ax=axs[2])
        sns.lineplot(data=dfData, x="rel_probability", y="count", ax=axs[3])
        # sns.histplot(data=dfData, x="rel_probability", kde=True, weights="count", ax=axs[3])
        plt.show()

        dataYaml = [mp.toDict() for mp in setData]
        breakpoint()

        out = OUTPUT_PATH / (problem_name + ".yaml")
        print(len(dataYaml), skipped)
        with open(out, "w") as file:
            yaml.dump(dataYaml, file, default_flow_style=None)


if __name__ == "__main__":
    main()
