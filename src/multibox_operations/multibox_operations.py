from typing import List

import torch


class Box:
    # a box
    lower: torch.Tensor
    upper: torch.Tensor

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def intersect(self, other):
        lower = torch.maximum(self.lower, other.lower)
        upper = torch.minimum(self.upper, other.upper)
        return Box(lower, upper)

    def arbitrary_intersection(self, other: list):
        lower = self.lower
        upper = self.upper
        for o in other:
            lower = torch.maximum(lower, o.lower)
            upper = torch.minimum(upper, o.upper)
        return Box(lower, upper)

    def is_empty(self):
        return torch.any(self.lower > self.upper)

    def area(self):
        if self.is_empty():
            return 0
        return torch.prod(self.upper - self.lower)


class Multibox:
    # a union of boxes
    boxes: List[Box]

    def __init__(self, boxes):
        self.boxes = boxes

    def intersect(self, others):
        """
        U_{i=1}^{n} B_i cap U_{j=1}^{m} B_j = U_{k=1}^{p} B_k
        where p <= n + m
        """
        new_boxes = []
        for box in self.boxes:
            for other in others.boxes:
                intersection, _, _ = box.intersect(other)
                # if the intersection is not empty
                if not intersection.is_empty():
                    new_boxes.append(intersection)
        return Multibox(new_boxes)

    def index_power_set(self):
        """
        Returns the power set of the index of the boxes
        """
        n = len(self.boxes)
        return [[j for j in range(n) if i & (1 << j)] for i in range(1 << n)]

    def area(self):
        """
        Uses inclusion-exclusion principle to calculate the area of the union of boxes
        Vol(U_{i=1}^{n} B_i) = Sum_{S subset {1, ..., n}} (-1)^{|S| + 1} Vol(cap_{i in S} B_i)
        """
        area = 0
        if len(self.boxes) == 0:
            return 0
        if len(self.boxes) == 1:
            return self.boxes[0].area()
        index_powerset = self.index_power_set()
        for indices in index_powerset:
            if len(indices) == 0:
                continue
            if len(indices) == 1:
                area += self.boxes[indices[0]].area()
                continue
            intersection = self.boxes[indices[0]]
            for i in range(1, len(indices)):
                intersection = intersection.intersect(self.boxes[indices[i]])
            area += (-1) ** (len(indices) + 1) * intersection.area()
        return area
