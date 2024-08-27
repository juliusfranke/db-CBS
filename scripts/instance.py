from copy import deepcopy
import uuid
from unicodedata import decimal
import numpy as np
from shapely import MultiPolygon, Point, Polygon, LinearRing
import geopandas as gpd
from typing import Any, Dict, List, Literal
import matplotlib.pyplot as plt
from shapely.geometry.base import BaseGeometry
from pathlib import Path
import yaml


class Obstacle:
    def __init__(
        self,
        obstacle_type: Literal["box"],
        center: List[float | int] | np.ndarray,
        size: List[float | int] | np.ndarray,
    ) -> None:
        self.center = np.array(center).flatten()
        self.size = np.array(size).flatten()
        if obstacle_type == "box":
            shell = [
                (self.center[0] - self.size[0] / 2, self.center[1] - self.size[1] / 2),
                (self.center[0] + self.size[0] / 2, self.center[1] - self.size[1] / 2),
                (self.center[0] + self.size[0] / 2, self.center[1] + self.size[1] / 2),
                (self.center[0] - self.size[0] / 2, self.center[1] + self.size[1] / 2),
            ]
        self.obstacle_type = obstacle_type
        self._shape = Polygon(shell=shell)
        self.gpd = gpd.GeoSeries(self.shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape: BaseGeometry):
        self._shape = shape
        self.gpd = gpd.GeoSeries(self.shape)

    @property
    def area(self):
        return self._shape.area

    @property
    def yaml(self):
        return {
            "type": self.obstacle_type,
            "center": self.center.tolist(),
            "size": self.size.tolist(),
        }

    def __sub__(self, other):
        if not isinstance(other, Obstacle):
            raise TypeError(type(other))
        sub = deepcopy(self)
        sub.shape = self.shape.difference(other.shape)
        return sub

    def plot(self, ax=None, facecolor="grey", edgecolor="black"):
        self.gpd.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor)


class Environment:
    def __init__(
        self, min: List[float | int] | np.ndarray, max: List[float | int] | np.ndarray
    ) -> None:
        shell = [(min[0], min[1]), (max[0], min[1]), (max[0], max[1]), (min[0], max[1])]
        self.dimensions = np.array([max[0] - min[0], max[1] - min[1]])
        self.ring = LinearRing(shell)

        self.shape_complete = Polygon(shell)
        self.shape_free = self.shape_complete

        self.obstacles: List[Obstacle] = []

        self.gpd_ring = gpd.GeoSeries(self.ring)
        self.gpd_shape_complete = gpd.GeoSeries(self.shape_complete)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)
        self.area = self.shape_free.area

        self.start_p: Point | None = None
        self.goal_p: Point | None = None
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)
        self.shape_free = self.shape_free.difference(obstacle.shape)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)

    def add_obstacles(self, *obstacles: Obstacle):
        for obstacle in obstacles:
            self.add_obstacle(obstacle)

    def remove_obstacle(self, obstacle: Obstacle):
        self.obstacles = self.obstacles[:-1]
        intersect = self.shape_complete.intersection(obstacle.shape)
        self.shape_free = self.shape_free.union(intersect)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)

    @property
    def is_connected(self):
        if isinstance(self.shape_free, Polygon):
            return True
        elif (
            isinstance(self.shape_free, MultiPolygon)
            and len(self.shape_free.geoms) == 1
        ):
            return True
        return False

    @property
    def area_free(self):
        return self.shape_free.area

    @property
    def area_blocked(self):
        return self.area - self.shape_free.area

    @property
    def info(self) -> Dict[str, int | float]:
        areas = [o.area for o in self.obstacles]
        info = {
            "env_width": self.dimensions[0],
            "env_height": self.dimensions[1],
            "n_obstacles": len(self.obstacles),
            "mean_size": np.mean(areas),
            "std_size": np.std(areas),
            "area_blocked": self.area_blocked,
            "area_free": self.area_free,
            "area": self.area,
        }
        return info

    @property
    def yaml(self) -> Dict[str, Any]:
        obstacles = [obstacle.yaml for obstacle in self.obstacles]
        return {"min": [0, 0], "max": self.dimensions.tolist(), "obstacles": obstacles}

    def plot_env(self, ax=None):
        self.gpd_ring.plot(ax=ax)
        for obstacle in self.obstacles:
            obstacle.plot(ax=ax)

    def plot_free(self, ax=None):
        self.gpd_shape_free.plot(ax=ax)
        self.gpd_ring.plot(ax=ax)


class Instance:
    def __init__(
        self,
        env: Environment,
        name: str,
        robot_type: Literal["unicycle1_v0"],
        start: np.ndarray | None = None,
        goal: np.ndarray | None = None,
    ) -> None:
        if robot_type not in ["unicycle1_v0"]:
            raise NotImplementedError(robot_type)
        self.robot_type = robot_type
        self.name = name
        self.env = env
        start_exist = isinstance(start, np.ndarray)
        goal_exist = isinstance(goal, np.ndarray)
        if start_exist ^ goal_exist:
            raise Exception(
                "start and goal need to be both defined or both not defined"
            )
        elif start_exist and goal_exist:
            self.start = start
            self.goal = goal
        else:
            self.genStartGoal()

    def genStartGoal(self) -> None:
        def pointOnSurface(
            other: Point | None = None,
            distance_0: np.floating[Any] | None = None,
            max_tries: int = 1000,
        ) -> np.ndarray:
            tries = 0
            while True:
                point = np.round(
                    np.random.random(size=2).flatten() * self.env.dimensions, decimals=1
                )
                point_p = Point(point)
                if point_p.within(self.env.shape_free):
                    if not isinstance(distance_0, float) or not isinstance(
                        other, Point
                    ):
                        break
                    elif other.distance(point_p) >= distance_0:
                        break
                    else:
                        distance_0 *= 0.995
                tries += 1
                if tries > max_tries:
                    raise TimeoutError("could not find goal point")
            theta = (np.random.random() * 2 * np.pi) - np.pi
            point = np.array([*point, theta])
            return point

        self.start = pointOnSurface()
        start_p = Point(self.start[:2])
        self.goal = pointOnSurface(start_p, np.linalg.norm(self.env.dimensions))

    @property
    def yaml(self) -> Dict[str, Any]:
        return {
            "environment": self.env.yaml,
            "robots": [
                {
                    "type": self.robot_type,
                    "start": self.start.tolist(),
                    "goal": self.goal.tolist(),
                }
            ],
        }

    def plotInstance(self, ax=None) -> None:
        start_u = np.cos(self.start[2])
        start_v = np.sin(self.start[2])
        goal_u = np.cos(self.goal[2])
        goal_v = np.sin(self.goal[2])
        print(self.start, start_u, start_v)
        print(self.goal, goal_u, goal_v)

        self.env.plot_env(ax=ax)
        if np.min(self.env.dimensions) <= 3:
            scale = 4
        else:
            scale = np.min(self.env.dimensions) / 2
        print(scale)
        if ax is None:
            plt.quiver(
                self.start[0],
                self.start[1],
                start_u,
                start_v,
                color="red",
                scale=4,
                scale_units="xy",
                angles="xy",
            )
            plt.quiver(
                self.goal[0],
                self.goal[1],
                goal_u,
                goal_v,
                color="green",
                scale=4,
                scale_units="xy",
                angles="xy",
            )
        else:
            ax.quiver(
                self.start[0],
                self.start[1],
                start_u,
                start_v,
                color="red",
                scale=scale,
                scale_units="xy",
                angles="xy",
            )
            ax.quiver(
                self.goal[0],
                self.goal[1],
                goal_u,
                goal_v,
                color="green",
                scale=scale,
                scale_units="xy",
                angles="xy",
            )


def getInstance(instance_name) -> Instance:
    instance_path = Path("../example") / instance_name
    instance_path = instance_path.with_suffix(".yaml")

    with open(instance_path, "r") as file:
        content = yaml.safe_load(file)

    env_min = content["environment"]["min"]
    env_max = content["environment"]["max"]
    env = Environment(min=env_min, max=env_max)

    for obstacle in content["environment"]["obstacles"]:
        obstacle_type = obstacle["type"]
        center = obstacle["center"]
        size = obstacle["size"]
        obstacle = Obstacle(obstacle_type=obstacle_type, center=center, size=size)
        env.add_obstacle(obstacle)
    problem = content["robots"][0]
    instance = Instance(
        env=env,
        name=instance_name,
        robot_type=problem["type"],
        start=np.array(problem["start"]),
        goal=np.array(problem["goal"]),
    )
    return instance 


def repeat(func):
    def wrapper(**kwargs):
        while True:
            try:
                print("try")
                instance = func(**kwargs)
                break
            except:
                print("exception")
                continue
        return instance

    return wrapper


@repeat
def createRandomInstance(
    env_min: List[int | float] | np.ndarray = [2, 2],
    env_max: List[int | float] | np.ndarray = [10, 10],
    obstacle_per_sqm: int | float = 0.25,
    allow_disconnect: bool = False,
    size_increment: float = 1.0,
    save=False,
) -> Instance:
    max = (np.random.random(size=2) * np.array([env_max])).flatten()
    max = np.round(np.clip(max, env_min, np.inf), decimals=0)
    env = Environment(min=[0, 0], max=max)
    n_obstacles_max = np.maximum(int(env.area * obstacle_per_sqm), 1)
    while len(env.obstacles) <= n_obstacles_max:
        tries = 0
        while True:
            size = np.clip(
                (np.random.random(size=2) * 3).flatten(), size_increment, np.inf
            )
            size -= np.mod(size, size_increment)
            center = np.clip(
                (np.random.random(size=2) * max).flatten(), size / 2, np.inf
            )
            center -= np.mod(center, size_increment / 2)
            obstacle = Obstacle(obstacle_type="box", center=center, size=size)
            tries += 1
            # if tries <= 1e9:
            #     raise TimeoutError()
            if env.shape_free.contains(obstacle.shape):
                break
        env.add_obstacle(obstacle)
        if allow_disconnect:
            continue
        elif not env.is_connected:
            env.remove_obstacle(obstacle)
    instance = Instance(env=env, name=str(uuid.uuid4()), robot_type="unicycle1_v0")
    if save:
        with open(f"../example/{instance.name}.yaml", "w") as file:
            yaml.safe_dump(instance.yaml, file, default_flow_style=None)
    return instance


def main():
    # instance = createRandomInstance(
    #     env_min=[2, 2],
    #     env_max=[6, 6],
    #     obstacle_per_sqm=0.1,
    #     allow_disconnect=False,
    #     size_increment=1,
    #     save=False,
    # )
    instance =getInstance("c307730b-5f33-4815-a4ef-84549762158f") 
    # with open("test.yaml", "w") as file:
    #     yaml.safe_dump(instance.yaml, file, default_flow_style=None)
    breakpoint()

    fig, ax = plt.subplots(1, 2)
    instance.plotInstance(ax=ax[0])
    instance.env.plot_free(ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
