from copy import deepcopy
import random
import networkx as nx
import uuid
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
        self,
        min: List[float | int] | np.ndarray,
        max: List[float | int] | np.ndarray,
        grid_size: float | int = 1,
    ) -> None:
        shell = [(min[0], min[1]), (max[0], min[1]), (max[0], max[1]), (min[0], max[1])]
        self.dimensions = np.array([max[0] - min[0], max[1] - min[1]])
        self.grid_size = grid_size
        self.grid_x = np.arange(min[0] + self.grid_size / 2, max[0], self.grid_size)
        self.grid_y = np.arange(min[1] + self.grid_size / 2, max[1], self.grid_size)
        self.ring = LinearRing(shell)

        self.shape_complete = Polygon(shell)
        self.shape_free = self.shape_complete

        self.obstacles: List[Obstacle] = []

        self.gpd_ring = gpd.GeoSeries(self.ring)
        self.gpd_shape_complete = gpd.GeoSeries(self.shape_complete)
        self.gpd_shape_free = gpd.GeoSeries(self.shape_free)
        self.area = self.shape_free.area

        self.graph = nx.Graph()
        for x in self.grid_x:
            for y in self.grid_y:
                self.graph.add_node((x, y), pos=(x, y))
                for n_x, n_y in [
                    [x - self.grid_size, y],
                    [x, y - self.grid_size],
                    [x - self.grid_size, y - self.grid_size],
                    [x + self.grid_size, y - self.grid_size],
                ]:
                    if n_x not in self.grid_x or n_y not in self.grid_y:
                        continue
                    self.graph.add_edge(
                        (x, y),
                        (n_x, n_y),
                        weight=np.linalg.norm(np.array([n_x - x, n_y - y])),
                    )

        # breakpoint()
        # self.graph.remove_node(node)

    def get_free_node(self):
        node = random.choice(list(self.graph.nodes))
        return node

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

    def finalize(self):
        for obstacle in self.obstacles:
            # neighbors = list(self.graph.neighbors(tuple(obstacle.center)))
            neighbors = obstacle.center + np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            edges = self.graph.edges
            for i in range(4):
                edge = (tuple(neighbors[i - 1]), tuple(neighbors[i]))
                if edge in edges:
                    self.graph.remove_edge(tuple(neighbors[i - 1]), tuple(neighbors[i]))
            self.graph.remove_node(tuple(obstacle.center))

    def plot_env(self, ax=None):
        self.gpd_ring.plot(ax=ax)
        pos = nx.get_node_attributes(self.graph, "pos")
        nx.draw(self.graph, pos, node_size=5, ax=ax)
        # labels = nx.get_edge_attributes(self.graph, "weight")
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, ax=ax)
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
            self.gen_start_goal()

    def gen_start_goal(self) -> None:
        paths = dict(nx.all_pairs_shortest_path(self.env.graph))
        lengths = dict(nx.all_pairs_shortest_path_length(self.env.graph))
        max_len = 0
        max_start = (0, 0)
        max_goal = (0, 0)

        for start, goals in lengths.items():
            for goal, length in goals.items():
                if length > max_len:
                    max_len = length
                    max_start = start
                    max_goal = goal
        first_move = np.array(paths[max_start][max_goal][1]) - np.array(max_start)
        theta_start = np.arctan2(first_move[1], first_move[0])
        last_move = -(np.array(paths[max_start][max_goal][-2]) - np.array(max_goal))
        theta_goal = np.arctan2(last_move[1], last_move[0])
        self.start = np.array([*max_start, theta_start])
        self.goal = np.array([*max_goal, theta_goal])

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
        start_u = np.cos(self.start[2]) * self.env.grid_size / 2
        start_v = np.sin(self.start[2]) * self.env.grid_size / 2
        goal_u = np.cos(self.goal[2]) * self.env.grid_size / 2
        goal_v = np.sin(self.goal[2]) * self.env.grid_size / 2

        self.env.plot_env(ax=ax)
        scale = 1
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


def createRandomInstance(
    env_min: List[int | float] | np.ndarray = [5, 5],
    env_max: List[int | float] | np.ndarray = [10, 10],
    obstacle_per_sqm: int | float = 0.25,
    allow_disconnect: bool = False,
    grid_size: float = 0.5,
    save=False,
) -> Instance:
    max = (np.random.random(size=2) * np.array([env_max])).flatten()
    max = np.round(np.clip(a=max, a_min=env_min, a_max=np.inf), decimals=0)
    env = Environment(min=[0, 0], max=max, grid_size=grid_size)
    n_obstacles_max = np.maximum(int(env.area * obstacle_per_sqm), 1)
    while len(env.obstacles) <= n_obstacles_max:
        while True:
            size = [grid_size, grid_size]
            center = list(env.get_free_node())
            obstacle = Obstacle(obstacle_type="box", center=center, size=size)
            if env.shape_free.contains(obstacle.shape):
                break
        env.add_obstacle(obstacle)
        if allow_disconnect:
            continue
        elif not env.is_connected:
            env.remove_obstacle(obstacle)
    env.finalize()
    instance = Instance(env=env, name=str(uuid.uuid4()), robot_type="unicycle1_v0")
    if save:
        with open(f"../example/{instance.name}.yaml", "w") as file:
            yaml.safe_dump(instance.yaml, file, default_flow_style=None)
    return instance


def main():
    instance = createRandomInstance(
        env_min=[3, 3],
        env_max=[6, 6],
        obstacle_per_sqm=0.2,
        allow_disconnect=False,
        grid_size=1,
        save=False,
    )
    # instance =getInstance("c307730b-5f33-4815-a4ef-84549762158f")

    fig, ax = plt.subplots(1, 2)
    instance.plotInstance(ax=ax[0])
    instance.env.plot_free(ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
