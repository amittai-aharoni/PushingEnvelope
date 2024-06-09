import torch


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
    def intersect(multiboxes_A, multiboxes_B, device, suspicious=True):
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
        return Multiboxes(
            torch.stack(mins, dim=1),
            torch.stack(maxs, dim=1),
            suspicious=suspicious,
            device=device,
        )

    @staticmethod
    def top(dim):
        """
        Returns a muiltibox that covers the entire space
        by having the minimum -inf and the maximum +inf
        """
        max = torch.full([dim], float(2))
        min = torch.full([dim], float(-2))
        return min, max

    @staticmethod
    def quasi_monte_carlo_area(multiboxes, points, device):
        """
        Computes a soft measure of how many points fall within the boundaries of multiboxes using
        a quasi-Monte Carlo method enhanced by sigmoid thresholding. Each point's inclusion is assessed
        in a probabilistic sense relative to each box in the multibox structure.

        Parameters:
        - multiboxes (NamedTuple): A named tuple with 'min' and 'max' tensors that define the
                                   minimum and maximum corners of boxes in a multidimensional space.
                                   The shape of 'min' and 'max' tensors should be [batch_size, boxes_amount, dim].
        - points (torch.Tensor): A tensor of points in the same dimensional space as the boxes.
                                 Shape should be [number_of_points, dim].
        - device (torch.device): The device (e.g., CPU or GPU) to which tensors
                                                should be transferred for calculation.

        Returns:
        - torch.Tensor: A tensor of shape [batch_size, number_of_points] representing the soft inclusion score
                        for each point, relative to the multiboxes, where each entry indicates the maximum
                        inclusion probability across all boxes for a given point in a given batch.

        This function expands the points tensor to match the dimensions of the multibox tensors, computes
        the sigmoid of differences between points and box boundaries, and processes these values to
        derive a probabilistic inclusion score for each point with respect to all boxes. The result is
        a tensor where higher values indicate a higher probability of a point being considered inside
        any of the boxes in the multibox.
        """
        batch_size = multiboxes.min.shape[0]
        boxes_amount = multiboxes.min.shape[1]
        dim = multiboxes.min.shape[2]
        number_of_points = points.shape[0]

        # For each multibox, we calculate the inclusion of each point
        points = (points.unsqueeze(1).expand(batch_size, number_of_points, 1, dim)).to(
            device
        )

        # for each dimension, if sigmoid > 0.5, then the point is in the bounds
        # of the box for that dimension
        # Hence the mean of the sigmoid > 0.5 if the point is in the box
        # For each multibox make a copy of the set of boxes for each point
        multibox_min_expanded = multiboxes.min.unsqueeze(1).expand(
            batch_size, number_of_points, boxes_amount, dim
        )
        multibox_max_expanded = multiboxes.max.unsqueeze(1).expand(
            batch_size, number_of_points, boxes_amount, dim
        )
        lower_bound = (points - multibox_min_expanded).sigmoid().unsqueeze(dim=4)
        upper_bound = (multibox_max_expanded - points).sigmoid().unsqueeze(dim=4)
        differences_cat = torch.cat([lower_bound, upper_bound], dim=4).mean(dim=4)
        differences_cat = differences_cat.mean(dim=3)
        differences_cat = differences_cat.max(dim=2).values
        soft_inclusion = differences_cat

        return soft_inclusion

    @staticmethod
    def get_existential(d_multiboxes, r_multiboxes, device, suspicious=True):
        """
        In description logic and ontology
        ∃R.C = π[R ⊓ π^(-1)(C)]
        This function computes the preimage of the d_multiboxes
        and intersects it with the r_multiboxes.
        Finally, it returns the projection of the intersection
        """
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
            r_multiboxes, d_pre_image_multiboxes, device, suspicious=suspicious
        )
        dim = intersection.min.shape[2]
        existential_multiboxes_min = intersection.min[:, :, : dim // 2]
        existential_multiboxes_max = intersection.max[:, :, : dim // 2]
        existential_multiboxes = Multiboxes(
            existential_multiboxes_min, existential_multiboxes_max
        )
        return existential_multiboxes
