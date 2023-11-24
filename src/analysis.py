import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Actor, Evaluator

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


actor_params = torch.load("params/actor_params", map_location=DEVICE)()
eval_params = torch.load("params/evaluator_params", map_location=DEVICE)()

fig, axes = plt.subplots(3,2, figsize=(20, 15))
actor_maps = actor_params["layer.weight"] 
evaluator_map = eval_params["layer.weight"] 

titles = [
        "Weights of the speed actuator",
        "Weights of the rotation actuator",
        "Weights for the control of noise in speed response",
        "Weights for the control of noise in rotation response",
        "Weights for the state evaluation",
        ]

for i in range(2):
    for j in range(2):
        curmap = actor_maps[j*2 + i]
        curmap = curmap.reshape(50, 80).T
        ax = axes[i, j]
        ax.set_title(titles[j*2 + i])
        hinton(curmap, ax=ax)

curmap = evaluator_map
curmap = curmap.reshape(50, 80).T
ax = axes[2, 0]
ax.set_title(titles[-1])
hinton(curmap, ax=ax)

ax = axes[2, 1]
ax.set_axis_off()

fig.tight_layout()
fig.savefig("analysis.png", dpi=400)
