import pytest
import torch

from src.multibox_operations.multibox_operations import Multibox, Box


@pytest.mark.parametrize(
    "boxes, expected",
    [
        # test disjoint case
        (
            [
                Box(torch.tensor([-1]), torch.tensor([1])),
                Box(torch.tensor([2]), torch.tensor([4])),
            ],
            4,
        ),
        (
            [
                Box(torch.tensor([-1]), torch.tensor([1])),
                Box(torch.tensor([1]), torch.tensor([3])),
            ],
            4,
        ),
        # test overlapping case
        (
            [
                Box(torch.tensor([-1]), torch.tensor([1])),
                Box(torch.tensor([0]), torch.tensor([1])),
            ],
            2,
        ),
        # three boxes
        (
            [
                Box(torch.tensor([-1]), torch.tensor([1])),
                Box(torch.tensor([0]), torch.tensor([2])),
                Box(torch.tensor([4]), torch.tensor([6])),
            ],
            5,
        ),
        ([], 0),
    ],
)
def test_area(boxes, expected):
    multibox = Multibox(boxes)
    assert multibox.area() == expected
