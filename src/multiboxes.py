import torch


class Multiboxes:
    def __init__(self, min, max):
        # min and max are tensors of shape (b, n, d)
        # where b is the batch size,
        # n is the number of boxes, and d is the dimensionality
        self.min = min
        self.max = max

    def to(self, device):
        return Multiboxes(self.min.to(device), self.max.to(device))

    @staticmethod
    def intersect(multiboxes_A, multiboxes_B):
        """
        Computes the intersection of two sets of boxes
        """
        mins = []
        maxs = []
        for index_A in range(multiboxes_A.min.shape[1]):
            for index_B in range(multiboxes_B.min.shape[1]):
                min_A = multiboxes_A.min[:, index_A]
                max_A = multiboxes_A.max[:, index_A]
                min_B = multiboxes_B.min[:, index_B]
                max_B = multiboxes_B.max[:, index_B]
                min = torch.max(torch.stack([min_A, min_B]), dim=0).values
                max = torch.min(torch.stack([max_A, max_B]), dim=0).values
                mins.append(min)
                maxs.append(max)
        return Multiboxes(torch.stack(mins, dim=1), torch.stack(maxs, dim=1))

    @staticmethod
    def union(multiboxes_A, multiboxes_B):
        """
        Computes the union of two sets of boxes
        """
        min = torch.cat([multiboxes_A.min, multiboxes_B.min], dim=1)
        max = torch.cat([multiboxes_A.max, multiboxes_B.max], dim=1)
        return Multiboxes(min, max)

    @staticmethod
    def top(dim):
        """
        Returns a muiltibox that covers the entire space
        by having the minimum -inf and the maximum +inf
        """
        max = torch.full([dim], float("inf"))
        min = torch.full([dim], float("-inf"))
        return min, max

    @staticmethod
    def index_power_set(n, device):
        """
        Returns the power set of the index of the boxes
        """
        return [
            torch.tensor([j for j in range(n) if i & (1 << j)]).to(device)
            for i in range(1 << n)
        ]

    @staticmethod
    def sign(indices):
        return len(indices) % 2 * 2 - 1

    @staticmethod
    def area(multiboxes, device):
        n = multiboxes.min.shape[1]
        power_set = Multiboxes.index_power_set(n, device)
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

    @staticmethod
    def get_existential(d_multiboxes, r_multiboxes, device):
        batch_size = d_multiboxes.min.shape[0]
        boxes_amount = d_multiboxes.min.shape[1]
        dim = d_multiboxes.min.shape[2]
        top_min, top_max = Multiboxes.top(dim)
        top_min.to(device)
        top_max.to(device)
        top_min = top_min.view(1, 1, dim).repeat(batch_size, boxes_amount, 1)
        top_max = top_max.view(1, 1, dim).repeat(batch_size, boxes_amount, 1)
        d_pre_image_min = torch.concat((top_min, d_multiboxes.min), dim=2)
        d_pre_image_max = torch.concat((top_max, d_multiboxes.max), dim=2)
        d_pre_image_multiboxes = Multiboxes(d_pre_image_min, d_pre_image_max)
        intersection = Multiboxes.intersect(r_multiboxes, d_pre_image_multiboxes)
        existential_multiboxes = intersection[:, :, : dim // 2]
        return existential_multiboxes
