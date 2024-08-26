from copy import deepcopy
from unicodedata import decimal
import numpy as np
from shapely import MultiPolygon, Polygon, LinearRing
import geopandas as gpd
from typing import Dict, List, Literal
import matplotlib.pyplot as plt
from shapely.geometry.base import BaseGeometry
from pathlib import Path


class Obstacle:
    def __init__(
        self,
        obstacle_type: Literal["box"],
        center: List[float | int] | np.ndarray,
        size: List[float | int] | np.ndarray,
    ) -> None:
        if obstacle_type == "box":
            shell = [
                (center[0] - size[0] / 2, center[1] - size[1] / 2),
                (center[0] + size[0] / 2, center[1] - size[1] / 2),
                (center[0] + size[0] / 2, center[1] + size[1] / 2),
                (center[0] - size[0] / 2, center[1] + size[1] / 2),
            ]
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

    def __sub__(self, other):
        if not isinstance(other, Obstacle):
            raise TypeError(type(other))
        sub = deepcopy(self)
        sub.shape = self.shape.difference(other.shape)
        return sub

    def plot(self, ax=None, facecolor="grey"):
        self.gpd.plot(ax=ax, facecolor=facecolor)


class Environment:
    def __init__(
        self, min: List[float | int] | np.ndarray, max: List[float | int] | np.ndarray
    ) -> None:
        shell = [(min[0], min[1]), (max[0], min[1]), (max[0], max[1]), (min[0], max[1])]
        self.dimensions = (max[0] - min[0], max[1] - min[1])
        self.ring = LinearRing(shell)

        self.shape_complete = Polygon(shell)
        self.shape_free = self.shape_complete

        self.obstacles: List[Obstacle] = []

        self.gpd_ring = gpd.GeoSeries(self.ring)
        self.gpd_shape_complete = gpd.GeoSeries(self.shape_complete)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)
        self.area = self.shape_free.area

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)
        self.shape_free = self.shape_free.difference(obstacle.shape)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)

    def add_obstacles(self, *obstacles: Obstacle):
        for obstacle in obstacles:
            self.add_obstacle(obstacle)

    def remove_obstacle(self, obstacle: Obstacle):
        self.obstacles.remove(obstacle)
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

    def plot_env(self, ax=None):
        self.gpd_ring.plot(ax=ax)
        for obstacle in self.obstacles:
            obstacle.plot(ax=ax)

    def plot_free(self, ax=None):
        self.gpd_shape_free.plot(ax=ax)


def createEnvironment(instance) -> Environment:
    instance_path = Path("../example") / instance
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
    return env


def createRandomEnvironment(
    env_max: List[int | float] = [10, 10],
    obstacle_per_sqm: int|float = 0.3,
    allow_disconnect: bool = False,
) -> Environment:
    max = (np.random.random(size=2) * np.array([env_max])).flatten()
    max = np.round(np.clip(max,2,np.inf)*2)/2
    env = Environment(min=[0, 0], max=max)
    n_obstacles_max = np.maximum(int(env.area * obstacle_per_sqm), 1)
    while len(env.obstacles) <= n_obstacles_max:
        if np.random.random() < (len(env.obstacles)/n_obstacles_max)**2:
            break
        while True:
            center = (np.random.random(size=2) * max).flatten()
            size = (np.random.random(size=2) * max / 2).flatten()
            size = np.clip(size,0.5,10)
            obstacle = Obstacle(obstacle_type="box", center=center, size=size)
            # if env.shape_complete.contains(obstacle.shape):
            #     break
            if obstacle.shape.within(env.shape_complete):
                break
        env.add_obstacle(obstacle)
        if allow_disconnect:
            continue
        elif not env.is_connected:
            env.remove_obstacle(obstacle)
    return env


def main():
    # env = Environment(min=[0, 0], max=[6, 6])
    # ob0 = Obstacle(obstacle_type="box", center=[1, 1], size=[1, 1])
    # ob00 = Obstacle(obstacle_type="box", center=[1, 1], size=[0.5, 0.5])
    # ob1 = Obstacle(obstacle_type="box", center=[2, 2], size=[1, 1])
    # ob2 = Obstacle(obstacle_type="box", center=[4, 1], size=[1, 1])
    # ob3 = Obstacle(obstacle_type="box", center=[1, 5], size=[1, 1])
    # ob = ob0 - ob00
    #
    # env.add_obstacles(ob, ob1, ob2, ob3)
    # print(env.is_connected(), env.area, env.area_free, env.area_blocked)
    # fix, ax = plt.subplots(1, 2)
    # env.plot_env(ax=ax[0])
    # env.plot_free(ax=ax[1])
    # plt.show()
    # env.remove_obstacle(ob)
    # print(env.is_connected(), env.area, env.area_free, env.area_blocked)
    env = createRandomEnvironment(obstacle_per_sqm=0.2,allow_disconnect=True)

    fix, ax = plt.subplots(1, 2)
    env.plot_env(ax=ax[0])
    env.plot_free(ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
