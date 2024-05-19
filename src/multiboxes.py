import torch


class Multiboxes:
    def __init__(self, min, max):
        # min and max are tensors of shape (b, n, d)
        # where b is the batch size,
        # n is the number of boxes, and d is the dimensionality
        self.min = min
        self.max = max

    @staticmethod
    def index_power_set(n):
        """
        Returns the power set of the index of the boxes
        """
        return [
            torch.tensor([j for j in range(n) if i & (1 << j)]) for i in range(1 << n)
        ]

    @staticmethod
    def sign(indices):
        return len(indices) % 2 * 2 - 1

    @staticmethod
    def area(multiboxes):
        n = len(multiboxes.min[0])
        power_set = Multiboxes.index_power_set(n)
        summands = []
        for indices in power_set:
            if len(indices) == 0:
                continue
            # min and max are tensors of shape (b, | S |, d) where | S | is the number of boxes in the subset
            min = torch.index_select(multiboxes.min, 1, indices)
            max = torch.index_select(multiboxes.max, 1, indices)

            min = torch.max(min, dim=1).values
            max = torch.min(max, dim=1).values
            diff = max - min
            diff = torch.max(diff, torch.zeros_like(diff))
            areas = torch.prod(diff, dim=1)
            # sign = (|S| + 1 % 2) * 2 - 1
            sign = Multiboxes.sign(indices)
            summands.append(sign * areas)
        summands = torch.stack(summands, dim=1)
        return torch.sum(summands, dim=1)
