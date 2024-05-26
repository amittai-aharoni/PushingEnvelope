import itertools
from collections import defaultdict

import torch
from tqdm import tqdm

from src.config import MONTE_CARLO_SAMPLES

# Read powerset_lookup.csv
POWERSETS_DICT = defaultdict(list)
with open("powerset_lookup.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        POWERSETS_DICT[int(line[0])] = eval(line[1])


class Multiboxes:
    def __init__(self, min, max, suspicious=False, device=""):
        """
        Initializes a set of multiboxes. A multibox is a set of boxes in a d-dimensional space.
        Each box is defined by a minimum and a maximum point.

        Parameters:
        min (torch.Tensor): A tensor of shape (b, n, d) where b is the batch size,
        n is the number of boxes, and d is the dimensionality.
        max (torch.Tensor): A tensor of shape (b, n, d) where b is the batch size,
        n is the number of boxes, and d is the dimensionality.
        suspicious (bool): A flag that indicates whether we might have boxes
        with negative volume.
        In this case, we replace the boxes with negative volume with 0 volume boxes.
        We then reduce the number of boxes to the maximum number of boxes
        that have positive volume.

        """
        if not suspicious:
            self.min = min
            self.max = max
        if suspicious:
            diff = max - min
            result = ~(diff < 0).any(dim=2)
            # Get the sorting indices
            _, indices = torch.sort(result.float(), dim=1, descending=True)
            # Use the indices to sort max, min, and result
            max_sorted = max.gather(1, indices.unsqueeze(-1).expand_as(max))
            min_sorted = min.gather(1, indices.unsqueeze(-1).expand_as(min))
            result_sorted = result.gather(1, indices)
            results_unsqueezed = result_sorted.unsqueeze(-1)
            dim = min.shape[2]
            # Create a tensor of 0 vectors of the same shape as min and max
            min_vector = torch.full([dim], float("inf")).to(device)
            max_vector = torch.full([dim], float("-inf")).to(device)
            # Substitute vectors in min and max that correspond to False in result with 0 vectors
            min = torch.where(results_unsqueezed, min_sorted, min_vector)
            max = torch.where(results_unsqueezed, max_sorted, max_vector)
            count = result.sum(dim=1)
            max_count = count.max()
            self.min = min[:, :max_count]
            self.max = max[:, :max_count]

    def to(self, device):
        return Multiboxes(self.min.to(device), self.max.to(device))

    @staticmethod
    def intersect(multiboxes_A, multiboxes_B, device):
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
                # if the number of boxes is different, we need to pad the smaller one with float("-inf") and float("inf")
                # if min_A.shape[1] < min_B.shape[1]:
                #     min_A = torch.cat([min_A, torch.full([min_A.shape[0], min_B.shape[1] - min_A.shape[1], min_A.shape[2]], float("-inf")).to(device)], dim=1)
                #     max_A = torch.cat([max_A, torch.full([max_A.shape[0], max_B.shape[1] - max_A.shape[1], max_A.shape[2]], float("inf")).to(device)], dim=1)
                # if min_B.shape[1] < min_A.shape[1]:
                #     min_B = torch.cat([min_B, torch.full([min_B.shape[0], min_A.shape[1] - min_B.shape[1], min_B.shape[2]], float("-inf")).to(device)], dim=1)
                #     max_B = torch.cat([max_B, torch.full([max_B.shape[0], max_A.shape[1] - max_B.shape[1], max_B.shape[2]], float("inf")).to(device)], dim=1)
                min = torch.max(torch.stack([min_A, min_B]), dim=0).values
                max = torch.min(torch.stack([max_A, max_B]), dim=0).values
                mins.append(min)
                maxs.append(max)
        return Multiboxes(
            torch.stack(mins, dim=1),
            torch.stack(maxs, dim=1),
            suspicious=True,
            device=device,
        )

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
    def index_power_set(n):
        """
        Returns the power set of the index of the boxes
        """
        print("Generating Power Set of size: ", n)
        return itertools.product([0, 1], repeat=n)

    @staticmethod
    def sign(indices):
        count = torch.sum(indices)
        return count % 2 * 2 - 1

    @staticmethod
    def area(multiboxes, device):
        n = multiboxes.min.shape[1]
        if n == 0:
            return torch.zeros(multiboxes.min.shape[0]).to(device)
        cache_power_set = False
        if n in POWERSETS_DICT:
            print("Using cached power set of size: ", n)
            power_set = POWERSETS_DICT[n]
        else:
            power_set = Multiboxes.index_power_set(n)
            cache_power_set = True
            cache_list = []
        mins = []
        maxs = []
        signs = []
        for indices in tqdm(power_set, desc="Computing Areas", total=2**n):
            if cache_power_set:
                cache_list.append(indices)
            indices = torch.tensor(indices).to(device)
            if len(indices) == 0:
                continue
            # min and max are tensors of shape (b, | S |, d) where | S | is the number of boxes in the subset
            min = torch.index_select(multiboxes.min, 1, indices)
            max = torch.index_select(multiboxes.max, 1, indices)

            min = torch.max(min, dim=1).values
            max = torch.min(max, dim=1).values
            signs.append(Multiboxes.sign(indices))
            mins.append(min)
            maxs.append(max)
        min = torch.stack(mins, dim=1)
        max = torch.stack(maxs, dim=1)
        diff = max - min
        diff = torch.max(diff, torch.zeros_like(diff))
        areas = torch.prod(diff, dim=2)
        # sign = (|S| + 1 % 2) * 2 - 1
        signs = torch.tensor(signs).to(device)
        areas = areas * signs
        if cache_power_set:
            POWERSETS_DICT[n] = cache_list
            # write to powerset_lookup.csv
            with open("powerset_lookup.csv", "a") as f:
                f.write(f"{n}\t{str(cache_list)}\n")
        area = torch.sum(areas, dim=1)
        return area

    @staticmethod
    def monte_carlo_area(multiboxes1, multiboxes2, device):
        # Generate random points
        n = MONTE_CARLO_SAMPLES
        batch_size = multiboxes1.min.shape[0]
        boxes_amount_1 = multiboxes1.min.shape[1]
        boxes_amount_2 = multiboxes2.min.shape[1]
        dim = multiboxes1.min.shape[2]
        minimum_value = torch.min(
            multiboxes1.min.min(dim=1).values, multiboxes2.min.min(dim=1).values
        )
        maximum_value = torch.max(
            multiboxes1.max.max(dim=1).values, multiboxes2.max.max(dim=1).values
        )

        random_points = torch.rand(batch_size, n, dim).to(device) * (
            maximum_value.unsqueeze(1) - minimum_value.unsqueeze(1)
        ) + minimum_value.unsqueeze(1)
        random_points_unsqueezed = random_points.unsqueeze(2)
        random_points_unsqueezed = random_points_unsqueezed.to(device)

        multiboxes = [
            {
                "min": multiboxes1.min,
                "max": multiboxes1.max,
                "boxes_amount": boxes_amount_1,
            },
            {
                "min": multiboxes2.min,
                "max": multiboxes2.max,
                "boxes_amount": boxes_amount_2,
            },
        ]

        for i in range(2):
            multibox = multiboxes[i]
            multibox_min_expanded = (
                multibox["min"]
                .unsqueeze(1)
                .expand(batch_size, n, multibox["boxes_amount"], dim)
            )
            multibox_max_expanded = (
                multibox["max"]
                .unsqueeze(1)
                .expand(batch_size, n, multibox["boxes_amount"], dim)
            )

            differences_min = random_points_unsqueezed - multibox_min_expanded
            differences_max = multibox_max_expanded - random_points_unsqueezed

            differences_min_sigmoide = torch.sigmoid(differences_min)
            differences_max_sigmoide = torch.sigmoid(differences_max)

            # collapse the dimensions of every box to get a single value
            differences_min_sigmoide_mean = differences_min_sigmoide.mean(dim=3)
            differences_max_sigmoide_mean = differences_max_sigmoide.mean(dim=3)

            # a point is included in a box if the sigmoid of the max and the min differences is greater than 0.5
            soft_include = torch.min(
                torch.stack(
                    [differences_min_sigmoide_mean, differences_max_sigmoide_mean],
                    dim=3,
                ),
                dim=3,
            ).values
            # a point is included in a multibox if it is included in at least one of the boxes
            soft_include_multibox = soft_include.max(dim=2).values
            multiboxes[i]["soft_include"] = soft_include_multibox

        return [multiboxes[0]["soft_include"], multiboxes[1]["soft_include"]]

    @staticmethod
    def get_existential(d_multiboxes, r_multiboxes, device):
        batch_size = d_multiboxes.min.shape[0]
        boxes_amount = d_multiboxes.min.shape[1]
        dim = d_multiboxes.min.shape[2]
        top_min, top_max = Multiboxes.top(dim)
        top_min.to(device)
        top_max.to(device)
        top_min = top_min.view(1, 1, dim).repeat(batch_size, boxes_amount, 1).to(device)
        top_max = top_max.view(1, 1, dim).repeat(batch_size, boxes_amount, 1).to(device)
        d_pre_image_min = torch.concat((top_min, d_multiboxes.min), dim=2)
        d_pre_image_max = torch.concat((top_max, d_multiboxes.max), dim=2)
        d_pre_image_multiboxes = Multiboxes(d_pre_image_min, d_pre_image_max)
        intersection = Multiboxes.intersect(
            r_multiboxes, d_pre_image_multiboxes, device
        )
        dim = intersection.min.shape[2]
        existential_multiboxes_min = intersection.min[:, :, : dim // 2]
        existential_multiboxes_max = intersection.max[:, :, : dim // 2]
        existential_multiboxes = Multiboxes(
            existential_multiboxes_min, existential_multiboxes_max
        )
        return existential_multiboxes
