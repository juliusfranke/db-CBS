# db-CBS: Discontinuity-Bounded Conflict-Based Search for Multi-Robot Kinodynamic Motion Planning
Db-CBS is a multi-robot kinodynamic motion planner that enables a team of robots with different dynamics, actuation limits, and shapes to reach their goals in challenging environments.
It solves this problem by combining the Multi-Agent Path Finding (MAPF) optimal solver Conflict-Based Search (CBS), the single-robot kinodynamic motion planner discontinuity-bounded A* (db-A*), and non- linear trajectory optimization.

Paper on [arXiv](https://arxiv.org/abs/2309.16445) and [Video](https://youtu.be/1mglNQOmOLE) are available.  


<img align="center" src="https://github.com/IMRCLab/db-CBS/assets/70643834/d371f288-00ec-443b-a5bc-8a79425fde0b" width="70%"/>



<!-- Robot dynamics such as unicycle, $2^{nd}$ order unicycle, double integrator, and car with trailer are implemented in this repository.  -->

## Get primitives

The primitives are on the TUB cloud. The easiest is to use a symlink:

```
ln -s /home/${USER}/tubCloud/projects/db-cbs/motions motions
```

Alternatively, download a copy

```
wget https://tubcloud.tu-berlin.de/s/CijbRaJadf6JwH3/download
unzip download
rm download
```

## Building

Tested on Ubuntu 22.04.

```
mkdir buildRelease
cd buildRelease
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/openrobots/" ..
make -j
```

## Running

```
cd buildRelease
python3 ../scripts/benchmark.py
```

## ROS
ROS2 workspace needs to have [crazyswarm2](https://github.com/IMRCLab/crazyswarm2). 

**Set Up**

Add a symlink of dbcbs_ros to your ROS2 workspace

```
ln <PATH-TO>/dbcbs_ros <PATH-TO>ros2_ws/src/ -s
```

**Build** 

```
colcon build --symlink-install 
```

**Usage**

Note that all configuration files are the ones used in cvmrs_ros/config. This allows to commit those files without changing the default value of the (public) crazyswarm2 repository. 

```
ros2 launch dbcbs_ros launch.py
```

and in a separate terminal

```
ros2 run dbcbs_ros multi_trajectory