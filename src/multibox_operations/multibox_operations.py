import torch

from typing import List
from mpmath import mp

mp.dps = 50  # set the precision


class Box:
    # a box
    center: torch.Tensor
    offsets: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def __init__(self, center=None, offsets=None, lower=None, upper=None):
        if lower is None and upper is None:
            self.center = center
            self.offsets = offsets
            self.lower = center - offsets
            self.upper = center + offsets
        if center is None and offsets is None:
            self.lower = lower
            self.upper = upper
            self.center = (lower + upper) / 2
            self.offsets = (upper - lower) / 2

    def intersect(self, other):
        lower = torch.maximum(self.lower, other.lower)
        upper = torch.minimum(self.upper, other.upper)
        return Box(lower=lower, upper=upper)

    def arbitrary_intersection(self, other: list):
        lower = self.lower
        upper = self.upper
        for o in other:
            lower = torch.maximum(lower, o.lower)
            upper = torch.minimum(upper, o.upper)
        return Box(lower=lower, upper=upper)

    def is_empty(self):
        return torch.any(self.lower > self.upper)

    # def area(self):
    #     if self.is_empty():
    #         return 0.
    #     delta = self.upper - self.lower
    #     # convert to mp.mpf to get higher precision
    #     delta = [mp.mpf(d.item()) for d in delta]
    #     area = mp.fprod(delta)
    #     return area

    def area(self):
        if self.is_empty():
            return torch.tensor(0.0)
        delta = self.upper - self.lower
        area = torch.prod(delta)
        return area

    def log_area(self):
        if self.is_empty():
            return 0.0
        delta = self.upper - self.lower
        log_prod = torch.sum(torch.log(delta))
        return log_prod

    def concat(self, other):
        """
        Create a new box that contains both boxes
        """
        center = torch.cat((self.center, other.center))
        offsets = torch.cat((self.offsets, other.offsets))
        return Box(center=center, offsets=offsets)

    def project1(self, dim=None):
        """
        Project the box to the first half of the dimensions
        """
        if dim is None:
            dim = int(len(self.center) / 2)
        center = self.center[:dim]
        offsets = self.offsets[:dim]
        return Box(center=center, offsets=offsets)

    def project2(self, dim=None):
        """
        Project the box to the second half of the dimensions
        """
        if dim is None:
            dim = int(len(self.center) / 2)
        center = self.center[dim:]
        offsets = self.offsets[dim:]
        return Box(center=center, offsets=offsets)

    @staticmethod
    def top_box(dim=int):
        """
        Returns a box that covers the entire space
        by having center at the origin and offsets of infinity
        """
        zeros = torch.zeros(dim)
        inf = torch.full((dim,), float("inf"))
        return Box(center=zeros, offsets=inf)

    def to(self, device):
        self.center = self.center.to(device)
        self.offsets = self.offsets.to(device)
        self.lower = self.lower.to(device)
        self.upper = self.upper.to(device)
        return self


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
                intersection = box.intersect(other)
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
            return 0.0
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
        # area = self.scale_area(area)
        return area

    def scale_area(self, area):
        # Scale the number
        scaled_num_mpf = area * mp.power(10, 318)
        # Convert the scaled number back to a PyTorch tensor
        scaled_num_tensor = torch.tensor(float(scaled_num_mpf), dtype=torch.float64)
        return scaled_num_tensor

    def project1(self, dim=None):
        """
        Project the multibox to the first half of the dimensions
        """
        return Multibox([box.project1(dim) for box in self.boxes])

    def project2(self, dim=None):
        """
        Project the multibox to the second half of the dimensions
        """
        return Multibox([box.project2(dim) for box in self.boxes])

    def concat(self, other):
        """
        Create a new multibox that contains both multiboxes
        """
        return Multibox(
            [box1.concat(box2) for box1, box2 in zip(self.boxes, other.boxes)]
        )
