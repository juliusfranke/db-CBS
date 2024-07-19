import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

def main():
    filename = Path("result_dbcbs.yaml")
    path = Path("../results/alcove_unicycle_single/db-cbs")
    # path = Path("../results/circle_unicycle_single/db-cbs")
    states_plot_old = np.array([])
    for folder in path.iterdir():
        filepath = folder / filename
        if not filepath.exists():
            print(f"{filepath} does not exists")
            continue
        with open(filepath, "r") as file:
            data = yaml.safe_load(file)
        states_plot = []
        for motion_primitive in data["motion_primitives"]:
            states = motion_primitive["states"]
            states_plot.extend(states)
            # actions = np.array(motion_primitive["actions"])
        states_plot = np.array(states_plot)
        # if states_plot_old.shape == states_plot.shape:
        #     dist = np.linalg.norm(states_plot_old - states_plot)
        #     print(dist)
        # else:
        #     print(states_plot_old.shape,states_plot.shape)
        # states_plot_old = states_plot
        plt.plot(states_plot[:,0], states_plot[:,1], label=f"{int(folder.name)}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
