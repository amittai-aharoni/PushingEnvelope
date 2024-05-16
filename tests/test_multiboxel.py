import pytest
import torch
from torch import tensor

from src.model.MultiBoxEL import MultiBoxEL
from src.multibox_operations.multibox_operations import Multibox, Box


@pytest.fixture(scope="session")
def model():
    return MultiBoxEL(
        device="cpu", embedding_dim=2, num_classes=3, num_boxes_per_class=2, num_roles=0
    )


@pytest.mark.parametrize(
    "embeddings, targets",
    [
        (
            [
                [[1, 1, 2, 1], [2, 1, 3, 1]],
                [
                    Multibox(
                        [
                            Box(torch.tensor([1]), torch.tensor([1])),
                            Box(torch.tensor([2]), torch.tensor([1])),
                        ]
                    ),
                    Multibox(
                        [
                            Box(torch.tensor([2]), torch.tensor([1])),
                            Box(torch.tensor([3]), torch.tensor([1])),
                        ]
                    ),
                ],
            ]
        )
    ],
)
def test_get_multiboxes(model, embeddings, targets):
    embeddings = torch.tensor(embeddings)
    multiboxes = model.get_multiboxes(embeddings)
    for i in range(len(multiboxes)):
        for j in range(len(multiboxes[i].boxes)):
            assert torch.allclose(
                multiboxes[i].boxes[j].center, targets[i].boxes[j].center
            )
            assert torch.allclose(
                multiboxes[i].boxes[j].offsets, targets[i].boxes[j].offsets
            )
            assert torch.allclose(
                multiboxes[i].boxes[j].lower, targets[i].boxes[j].lower
            )
            assert torch.allclose(
                multiboxes[i].boxes[j].upper, targets[i].boxes[j].upper
            )


@pytest.mark.parametrize(
    "boxes_A, boxes_B, target",
    [
        (
            [Box(torch.tensor([0]), torch.tensor([1]))],
            [Box(torch.tensor([0]), torch.tensor([1]))],
            0,
        ),
        (
            [Box(torch.tensor([0]), torch.tensor([2]))],
            [Box(torch.tensor([3]), torch.tensor([2]))],
            0.75,
        ),
        (
            [
                Box(torch.tensor([0]), torch.tensor([1])),
                Box(torch.tensor([1]), torch.tensor([1])),
            ],
            [Box(torch.tensor([0]), torch.tensor([2]))],
            0,
        ),
    ],
)
def test_inclusion_loss(model, boxes_A, boxes_B, target):
    multibox_A = Multibox(boxes_A)
    multibox_B = Multibox(boxes_B)
    dist = model.inclusion_loss([multibox_A], [multibox_B])
    assert dist == target


def test_area():
    """
    Test the area of the union of boxes.
    We test this method because underflow can occur when calculating
    the area of high-dimensional boxes.
    """
    boxes = []
    for i in range(4):
        center = torch.empty(50).uniform_(0, 0.2)
        offset = torch.empty(50).normal_(mean=0.5, std=0.1673)
        boxes.append(Box(center=center, offsets=torch.abs(offset)))
    multibox = Multibox(boxes)
    area = multibox.area()
    assert area > 0


def test_loss_function_gradient():
    torch.set_grad_enabled(True)
    data = {
        "top": tensor([0]),
        "role_chain": tensor([]),
        "class_ids": tensor([0, 1, 2]),
        "role_inclusion": tensor([]),
        "nf4": tensor([]),
        "disjoint": tensor([], dtype=torch.int64),
        "nf1": tensor([[1, 2]]),
        "nf2": tensor([]),
        "nf3": tensor([]),
        "nf3_neg0": tensor([]),
    }
    mutlibox_el = MultiBoxEL(
        device="cpu", embedding_dim=4, num_classes=3, num_boxes_per_class=1, num_roles=0
    )
    loss = mutlibox_el(data)
    print(loss)
    loss.backward()
