from dataclasses import dataclass

import torch


@dataclass
class Boxes:
    centers: torch.Tensor
    offsets: torch.Tensor

    def __init__(self, centers, offsets):
        self.centers = centers
        self.offsets = offsets

    @staticmethod
    def finite_intersection(boxes_list):
        """
        Cap_{i=1}^{n} B_i = B
        where B is a box
        """
        for i, boxes in enumerate(boxes_list):
            if i == 0:
                intersection = boxes
            else:
                intersection, _, _ = intersection.intersect(boxes)
        return intersection

    def intersect(self, others):
        lower = torch.maximum(
            self.centers - self.offsets, others.centers - others.offsets
        )
        upper = torch.minimum(
            self.centers + self.offsets, others.centers + others.offsets
        )
        centers = (lower + upper) / 2
        offsets = torch.abs(upper - lower) / 2
        return Boxes(centers, offsets), lower, upper

    def translate(self, directions):
        return Boxes(self.centers + directions, self.offsets)

    def area(self):
        return torch.prod(self.offsets * 2)

    def __getitem__(self, item):
        return Boxes(self.centers[item], self.offsets[item])
