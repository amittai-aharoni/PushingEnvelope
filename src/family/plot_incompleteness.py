from collections import defaultdict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import torch

from src.family.incompleteness_result_data import load_data
from src.model.loaded_models import LoadedModel

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams.update(
    {
        "font.size": 13,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)


def plot_box_boxsqel(center, offset, color, text, text_pos):
    low = center - offset
    sizes = 2 * offset
    lower = low.tolist()
    sizes = sizes.tolist()
    rect = mpatches.Rectangle(
        lower, sizes[0], sizes[1], fill=False, edgecolor=color, linewidth=1.5
    )
    plt.gca().add_patch(rect)

    if text_pos == "upper_right":
        coords = lower[0] + sizes[0] - 0.32, lower[1] + sizes[1] - 0.12
    elif text_pos == "lower_right":
        coords = lower[0] + sizes[0] - 0.35, lower[1] + 0.04
    else:  # default is upper left
        coords = lower[0] + 0.03, lower[1] + sizes[1] - 0.12

    plt.text(coords[0], coords[1], text, color=color)

def plot_box_multiboxel(min, max, color, text, text_pos):
    low = min
    offset = (max - min) / 2
    sizes = 2 * offset
    lower = low.tolist()
    sizes = sizes.tolist()
    rect = mpatches.Rectangle(
        lower, sizes[0], sizes[1], fill=False, edgecolor=color, linewidth=1.5
    )
    plt.gca().add_patch(rect)

    if text_pos == "upper_right":
        coords = lower[0] + sizes[0] - 0.32, lower[1] + sizes[1] - 0.12
    elif text_pos == "lower_right":
        coords = lower[0] + sizes[0] - 0.35, lower[1] + 0.04
    else:  # default is upper left
        coords = lower[0] + 0.03, lower[1] + sizes[1] - 0.12

    plt.text(coords[0], coords[1], text, color=color)

MODEL = "multiboxel"

data, classes, relations, individuals = load_data()
device = "cpu"
# model = LoadedModel.from_name("boxsqel", "out/boxsqel", 2, device, best=False)
if MODEL == "multiboxel":
    embedding_size = 4
else:
    embedding_size = 2
model = LoadedModel.from_name(MODEL, f"out_incompleteness/{MODEL}", embedding_size=embedding_size, device=device, best=False)
if MODEL == "multiboxel":
    class_boxes = model.get_multiboxes(model.class_embeds)
else:
    class_boxes = model.get_boxes(model.class_embeds)

loc = plticker.MultipleLocator(base=0.5)
plt.figure(figsize=(6, 4.5))
plt.gca().yaxis.set_major_locator(loc)

r = "#ff1f5b"
g = "#00cd6c"
b = "#009ade"
p = "#af58ba"
y = "#ffc61e"
o = "#f28522"
colors = [b, o, g, r, p, y]
pos_dict = defaultdict(lambda: "upper_left")
pos_dict["Parent"] = "upper_right"
pos_dict["Female"] = "lower_right"
for i, c in enumerate(["A", "B", "C", "AB", "BC", "AC"]):
    if MODEL == "multiboxel":
        max, min = class_boxes.max[classes[c]], class_boxes.min[classes[c]]
        boxes_amount = max.shape[0]
        for j in range(boxes_amount):
            plot_box_multiboxel(min[j], max[j], colors[i], c, pos_dict[c])
    else:
        center, offset = class_boxes.centers[classes[c]], class_boxes.offsets[classes[c]]
        plot_box_boxsqel(center, offset, colors[i], c, pos_dict[c])

plt.gca().autoscale()
plt.savefig(f"family_{MODEL}.pdf", bbox_inches="tight")
plt.show()


# model = LoadedModel.from_name("elbe", "out/elbe", 2, device, best=False)
# class_boxes = model.get_boxes(model.class_embeds)
# plt.figure(figsize=(6, 4.5))
# plt.gca().xaxis.set_major_locator(loc)
# plt.gca().yaxis.set_major_locator(loc)
#
#
# def plot_box_elbe(center, offset, color, text, text_pos):
#     low = center - offset
#     sizes = 2 * offset
#     lower = low.tolist()
#     sizes = sizes.tolist()
#     rect = mpatches.Rectangle(
#         lower, sizes[0], sizes[1], fill=False, edgecolor=color, linewidth=1.5
#     )
#     plt.gca().add_patch(rect)
#
#     if text_pos == "upper_right":
#         coords = lower[0] + sizes[0] - 0.17, lower[1] + sizes[1] - 0.07
#     elif text_pos == "lower_right":
#         coords = lower[0] + sizes[0] - 0.21, lower[1] + 0.04
#     elif text_pos == "upper_left":
#         coords = lower[0] + 0.02, lower[1] + sizes[1] - 0.12
#     elif isinstance(text_pos, tuple):
#         coords = text_pos
#
#     plt.text(coords[0], coords[1], text, color=color)
#
#
# pos_dict["Father"] = (-0.15, 0.51)  # type: ignore
# pos_dict["Parent"] = (0.4, 0.5)  # type: ignore
# pos_dict["Mother"] = (-0.03, -0.16)  # type: ignore
# for i, c in enumerate(["Parent", "Male", "Female", "Father", "Mother", "Child"]):
#     center, offset = class_boxes.centers[classes[c]], class_boxes.offsets[classes[c]]
#     plot_box_elbe(center, offset, colors[i], c, pos_dict[c])
#
# plt.gca().autoscale()
# plt.savefig("family_elbe.pdf", bbox_inches="tight")
# plt.show()
