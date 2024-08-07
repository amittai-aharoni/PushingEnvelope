import torch
import torch.optim as optim
from tqdm import trange

from src.family.incompleteness_result_data import load_data
from src.model.BoxSquaredEL import BoxSquaredEL
from src.model.MultiBoxEL import MultiBoxEL

torch.random.manual_seed(123)

MODEL = "multiboxel"

data, classes, relations, individuals = load_data()
device = "cpu"
if MODEL == "boxsqel":
    model = BoxSquaredEL(
        device,
        2,
        len(classes),
        len(relations),
        num_individuals=len(individuals),
        margin=0,
        reg_factor=1,
        num_neg=0,
        vis_loss=True,
    )
if MODEL == "multiboxel":
    model = MultiBoxEL(
        device,
        4,
        len(classes),
        num_boxes_per_class=3,
        num_roles=len(relations),
        num_individuals=len(individuals),
    )
# model = Elbe(device, classes,
# len(relations), embedding_dim=2,
# margin=0, vis_loss=True)
optimizer = optim.Adam(params=model.parameters(), lr=5e-2)
model = model.to(device)

model.train()
if MODEL == "multiboxel":
    # we discovered that multiboxel converges to its best solution after 200 epochs
    num_epochs = 35
if MODEL == "boxsqel":
    num_epochs = 300
pbar = trange(num_epochs)
for epoch in pbar:
    loss = model(data)
    pbar.set_postfix({"loss": f"{loss.item():.2f}"})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.save(f"out_incompleteness/{model.name}")
