import sys
from matplotlib import pyplot as plt

project_dir = Path(__file__).resolve().parent.parent
visualization_dir = Path(__file__).resolve().parent
figures_dir = project_dir / "visualization" / "figures"
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(visualization_dir))

import apples_oranges_pears as aop

if __name__ == "__main__":
    h2 = aop.forward_sigmoid(aop.h1, aop.output_biases, aop.output_weights)

    ax = plt.figure().add_subplot(projection="3d")

    ax.scatter(*h2[aop.y == 0].T, color=aop.colors[0])
    ax.scatter(*h2[aop.y == 1].T, color=aop.colors[1])
    ax.scatter(*h2[aop.y == 2].T, color=aop.colors[2])

    ax.set_xlabel("Appleness")
    ax.set_ylabel("Orangeness")
    ax.set_zlabel("Pearness")

    plt.show()
