import itertools
import os
import pickle
from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from tqdm import tqdm

from src.model.loaded_models import MultiBoxELLoadedModel
from src.multiboxes import Multiboxes
from src.config import EMBEDDING_BOUND, MONTE_CARLO_SAMPLES


class MultiBoxEL(nn.Module):
    def __init__(
        self,
        device: str,
        embedding_dim: int,
        num_classes: int,
        num_boxes_per_class: int,
        num_roles: int,
        num_individuals: int = 0,
        margin: float = 0,
        neg_dist: float = 2,
        reg_factor: float = 0.5,
        num_neg: int = 2,
        batch_size: int = 512,
        vis_loss: bool = False,
        wandb=None,
        monte_carlo_points=MONTE_CARLO_SAMPLES,
    ):
        """
        Args:
            device: device to run the model on
            embedding_dim: dimension of the embeddings
            num_classes: number of classes
            num_roles: number of roles
            num_individuals: number of individuals
            margin: margin for the loss
            neg_dist: control how unlikely the negative samples are made by the model
            reg_factor: a multiplier to prevent expressive role representation from overfitting
            num_neg: number of negative samples we generate per nf3 sample
            batch_size: batch size
            vis_loss: whether to use the visualisation loss
        """
        super(MultiBoxEL, self).__init__()

        self.name = "multiboxel"
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.margin = margin
        # TODO: negative sampling would be implemented later
        # self.neg_dist = neg_dist
        # self.num_neg = num_neg
        self.batch_size = batch_size
        self.vis_loss = vis_loss
        self.reg_factor = reg_factor

        self.negative_sampling = True

        # Embedding dimensions are doubled to store both the center and the offset
        self.class_embeds = self.init_embeddings(
            num_embeddings=self.num_classes,
            dim=embedding_dim,
            num_boxes_per_class=num_boxes_per_class,
        )
        # Individuals are represented as boxes
        self.individual_embeds = self.init_embeddings(
            self.num_individuals, embedding_dim, num_boxes_per_class=1
        )
        # Roles are represented as boxes in the relation space
        self.relation_embeds = self.init_embeddings(
            num_embeddings=self.num_roles,
            dim=embedding_dim * 2,
            num_boxes_per_class=num_boxes_per_class,
        )

        self.monte_carlo_points = self.init_stratified_points(
            num_points=monte_carlo_points,
            num_dimensions=embedding_dim // 2,
            num_strata_per_dim=2,
        )
        self.monte_carlo_multiplier = torch.tensor(4 / monte_carlo_points).to(device)

        self.wandb = wandb

    def init_stratified_points(
        self, min=-2, max=2, num_points=10000, num_dimensions=5, num_strata_per_dim=2
    ):
        """
        Generate stratified points in a hypercube.
        """
        pickle_file = "stratified_points.pkl"

        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                samples = pickle.load(f)
                return samples
        total_strata = num_strata_per_dim**num_dimensions
        points_per_strata = num_points // total_strata
        strata_bounds = torch.linspace(-2, 2, num_strata_per_dim + 1)
        samples = []
        iter = itertools.product(range(num_strata_per_dim), repeat=num_dimensions)
        # Loop over each combination of strata indices
        for strata in tqdm(
            iter, desc="Computing strata", total=num_strata_per_dim**num_dimensions
        ):
            # Calculate start and end bounds for each dimension
            starts = torch.tensor(
                [strata_bounds[strata[i]] for i in range(num_dimensions)]
            )
            ends = torch.tensor(
                [strata_bounds[strata[i] + 1] for i in range(num_dimensions)]
            )
            # Generate uniformly distributed points within the current stratum for each dimension
            stratum_samples = (
                torch.rand((points_per_strata, num_dimensions)) * (ends - starts)
            ) + starts
            samples.append(stratum_samples)
        samples = torch.cat(samples)
        # Shuffle the samples to remove any order introduced by the stratification
        samples = samples[torch.randperm(samples.shape[0])]
        with open(pickle_file, "wb") as f:
            pickle.dump(samples, f)
        return samples

    def init_embeddings(
        self,
        num_embeddings: int,
        dim: int,
        num_boxes_per_class: int = 1,
        min: int = EMBEDDING_BOUND * -1,
        max: int = EMBEDDING_BOUND,
        normalise: bool = False,
    ) -> nn.Embedding:
        """
        Initialise embeddings with uniform distribution and normalise them.

        Args:
            num_embeddings: size of the dictionary of embeddings
            num_boxes_per_class: number of boxes per class
            dim: size of each embedding vector
            min: minimum value of the uniform distribution
            max: maximum value of the uniform distribution
            normalise: whether to normalise the embeddings
        """
        if num_embeddings == 0:
            return None
        if dim < 2:
            raise ValueError(
                "Embedding dimension must be at least 2,"
                " to store both the center and the offset"
            )
        embeddings = nn.Embedding(num_embeddings, dim * num_boxes_per_class)
        embeddings.weight.requires_grad = True  # Set requires_grad to True
        # nn.init.uniform_(embeddings.weight, a=min, b=max)
        # Initialize embeddings
        for i in range(num_boxes_per_class):
            nn.init.uniform_(
                embeddings.weight[:, i * dim : i * dim + dim // 2], a=min, b=max
            )
            nn.init.normal_(
                embeddings.weight[:, i * dim + dim // 2 : (i + 1) * dim],
                mean=0.5,
                std=0.1673,
            ).clamp(min=0.01, max=0.99)
        if normalise:
            embeddings.weight.data /= torch.linalg.norm(
                embeddings.weight.data, axis=1
            ).reshape(-1, 1)
        return embeddings

    def get_multiboxes(self, embedding: torch.Tensor, is_relation=False) -> Multiboxes:
        embedding_dim = self.embedding_dim * 2 if is_relation else self.embedding_dim
        boxes_amount = embedding.shape[1] // embedding_dim
        minimums = []
        maximums = []
        # range over the rows of the embedding
        for i in range(embedding.shape[0]):
            boxes_minimum = []
            boxes_maximum = []
            # iterate over the number of boxes per class
            for j in range(boxes_amount):
                box_start = j * embedding_dim
                box_middle = box_start + embedding_dim // 2
                center = embedding[i, box_start:box_middle]
                offsets = torch.abs(
                    embedding[i, box_middle : box_start + embedding_dim]
                )
                boxes_minimum.append(center - offsets)
                boxes_maximum.append(center + offsets)
            boxes_minimum = torch.stack(boxes_minimum)
            boxes_maximum = torch.stack(boxes_maximum)
            minimums.append(boxes_minimum)
            maximums.append(boxes_maximum)
        minimums = torch.stack(minimums)
        maximums = torch.stack(maximums)
        multiboxes = Multiboxes(minimums, maximums)
        multiboxes = multiboxes.to(self.device)
        return multiboxes

    def get_class_multiboxes(
        self, nf_data: torch.Tensor, *indices: int
    ) -> Generator[Multiboxes, Any, None]:
        return (self.get_multiboxes(self.class_embeds(nf_data[:, i])) for i in indices)

    def get_relation_multiboxes(
        self, nf_data, *indices
    ) -> Generator[Multiboxes, Any, None]:
        if len(indices) == 1:
            index = indices[0]
            return self.get_multiboxes(
                self.relation_embeds(nf_data[:, index]), is_relation=True
            )
        return (
            self.get_multiboxes(self.relation_embeds(nf_data[:, i]), is_relation=True)
            for i in indices
        )

    def get_individual_multiboxes(
        self, nf_data, *indices
    ) -> Generator[Multiboxes, Any, None]:
        return (
            self.get_multiboxes(self.individual_embeds(nf_data[:, i])) for i in indices
        )

    def inclusion_loss(self, multiboxes1: Multiboxes, multiboxes2: Multiboxes):
        """
        Compute 1 - Area(B1 cap B2) / Area(B1)
        """
        multiboxes1.to(self.device)
        multiboxes2.to(self.device)
        multiboxes1_soft_inclusion = Multiboxes.quasi_monte_carlo_area(
            multiboxes1, self.monte_carlo_points, device=self.device
        )
        multiboxes2_soft_inclusion = Multiboxes.quasi_monte_carlo_area(
            multiboxes2, self.monte_carlo_points, device=self.device
        )
        multibox1_area_estimate = (
            multiboxes1_soft_inclusion - 0.5
        ).relu() * self.monte_carlo_multiplier

        intersection_inclusion = torch.cat(
            [
                multiboxes1_soft_inclusion.unsqueeze(-1),
                multiboxes2_soft_inclusion.unsqueeze(-1),
            ],
            dim=2,
        ).mean(dim=2)
        intersection_area_estimate = (
            intersection_inclusion - 0.5
        ).relu() * self.monte_carlo_multiplier

        loss = (1 - intersection_area_estimate / multibox1_area_estimate).mean()
        loss = torch.reshape(relu(loss), [-1, 1])
        return loss

    def disjoint_loss(self, multiboxes1: Multiboxes, multiboxes2: Multiboxes):
        """
        Compute 1 - Area(B1 cup B2) / Area(B1) + Area(B2)
        """
        multiboxes1.to(self.device)
        multiboxes2.to(self.device)

        multiboxes1_soft_inclusion = Multiboxes.quasi_monte_carlo_area(
            multiboxes1, self.monte_carlo_points, device=self.device
        )
        multiboxes2_soft_inclusion = Multiboxes.quasi_monte_carlo_area(
            multiboxes2, self.monte_carlo_points, device=self.device
        )
        multibox1_area_estimate = (multiboxes1_soft_inclusion - 0.5).relu().sum(
            dim=1
        ) * self.monte_carlo_multiplier
        multibox2_area_estimate = (multiboxes2_soft_inclusion - 0.5).relu().sum(
            dim=1
        ) * self.monte_carlo_multiplier

        union_inclusion = (
            torch.cat(
                [
                    multiboxes1_soft_inclusion.unsqueeze(-1),
                    multiboxes2_soft_inclusion.unsqueeze(-1),
                ],
                dim=2,
            )
            .max(dim=2)
            .values
        )
        union_area_estimate = (union_inclusion - 0.5).relu().sum(
            dim=1
        ) * self.monte_carlo_multiplier

        loss = (
            1
            - union_area_estimate / (multibox1_area_estimate + multibox2_area_estimate)
        ).mean()
        loss = torch.reshape(relu(loss), [-1, 1])
        return loss

    def neg_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(
            torch.linalg.norm(
                relu(diffs - boxes1.offsets - boxes2.offsets + self.margin), axis=1
            ),
            [-1, 1],
        )
        return dist

    def nf1_loss(self, nf1_data):
        """
        Loss of normal forms C ⊑ D
        """
        c_multiboxes, d_multiboxes = self.get_class_multiboxes(nf1_data, 0, 1)
        return self.inclusion_loss(c_multiboxes, d_multiboxes)

    def nf2_loss(self, nf2_data):
        """
        Loss of normal forms C ⊓ D ⊑ E
        """
        c_multiboxes, d_multiboxes, e_multiboxes = self.get_class_multiboxes(
            nf2_data, 0, 1, 2
        )
        intersection = Multiboxes.intersect(
            c_multiboxes, d_multiboxes, device=self.device
        )
        return self.inclusion_loss(intersection, e_multiboxes)

    def nf2_disjoint_loss(self, disjoint_data):
        """
        Loss of normal forms C ⊓ D ⊑ ⊥
        """
        c_multiboxes, d_multiboxes = self.get_class_multiboxes(disjoint_data, 0, 1)
        return self.disjoint_loss(c_multiboxes, d_multiboxes)

    def nf3_loss(self, nf3_data):
        """
        Loss of normal forms C ⊑ ∃R.D
        """
        c_multiboxes, d_multiboxes = self.get_class_multiboxes(nf3_data, 0, 2)
        r_multiboxes = self.get_relation_multiboxes(nf3_data, 1)
        existential_multiboxes = Multiboxes.get_existential(
            d_multiboxes, r_multiboxes, self.device
        )

        return self.inclusion_loss(c_multiboxes, existential_multiboxes)

    def nf3_neg_loss(self, neg_data):
        c_boxes, d_boxes = self.get_class_multiboxes(neg_data, 0, 2)
        c_bumps, d_bumps = self.bumps(neg_data[:, 0]), self.bumps(neg_data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(neg_data, 1)

        return self.neg_loss(c_boxes.translate(d_bumps), head_boxes), self.neg_loss(
            d_boxes.translate(c_bumps), tail_boxes
        )

    def role_assertion_loss(self, data):
        """
        Loss of role assertions {(a,b)} ⊑ R
        """
        a_multiboxes, b_multiboxes = self.get_individual_multiboxes(data, 1, 2)
        r_multiboxes = self.get_relation_multiboxes(data, 0)
        ab_min = torch.cat((a_multiboxes.min, b_multiboxes.min), dim=2)
        ab_max = torch.cat((a_multiboxes.max, b_multiboxes.max), dim=2)
        ab_multiboxes = Multiboxes(ab_min, ab_max)
        return self.inclusion_loss(ab_multiboxes, r_multiboxes)

    def role_assertion_neg_loss(self, neg_data):
        a_boxes, b_boxes = self.get_individual_multiboxes(neg_data, 1, 2)
        a_bumps, b_bumps = self.individual_bumps(neg_data[:, 1]), self.individual_bumps(
            neg_data[:, 2]
        )
        head_boxes, tail_boxes = self.get_relation_boxes(neg_data, 0)

        return self.neg_loss(a_boxes.translate(b_bumps), head_boxes), self.neg_loss(
            b_boxes.translate(a_bumps), tail_boxes
        )

    def concept_assertion_loss(self, data):
        (a_boxes,) = self.get_individual_multiboxes(data, 2)
        a_bumps = self.individual_bumps(data[:, 2])
        (c_boxes,) = self.get_class_multiboxes(data, 1)
        c_bumps = self.bumps(data[:, 1])
        head_boxes, tail_boxes = self.get_relation_multiboxes(data, 0)

        dist1 = self.inclusion_loss(a_boxes.translate(c_bumps), head_boxes)
        dist2 = self.inclusion_loss(c_boxes.translate(a_bumps), tail_boxes)
        return (dist1 + dist2) / 2

    def nf4_loss(self, nf4_data):
        """
        Loss of normal forms ∃R.C ⊑ D
        """
        c_multiboxes, d_multiboxes = self.get_class_multiboxes(nf4_data, 1, 2)
        r_multiboxes = self.get_relation_multiboxes(nf4_data, 0)
        existential_multiboxes = Multiboxes.get_existential(
            c_multiboxes, r_multiboxes, self.device
        )

        return self.inclusion_loss(existential_multiboxes, d_multiboxes)

    def role_inclusion_loss(self, data):
        r1_multiboxes, r2_multiboxes = self.get_relation_multiboxes(data, 0, 1)
        return self.inclusion_loss(r1_multiboxes, r2_multiboxes)

    def role_chain_loss(self, data):
        """
        Loss of role chains R1 o R2 ⊑ R3
        """
        r1_multiboxes, r2_multiboxes, s_multiboxes = self.get_relation_multiboxes(
            data, 0, 1, 2
        )
        dim = r1_multiboxes.min.shape[2]

        r1_heads_min = r1_multiboxes.min[:, :, : dim // 2]
        r1_heads_max = r1_multiboxes.max[:, :, : dim // 2]
        r1_heads = Multiboxes(r1_heads_min, r1_heads_max)
        r2_tails_min = r2_multiboxes.min[:, :, dim // 2 :]
        r2_tails_max = r2_multiboxes.max[:, :, dim // 2 :]
        r2_tails = Multiboxes(r2_tails_min, r2_tails_max)
        s_heads_min = s_multiboxes.min[:, :, : dim // 2]
        s_heads_max = s_multiboxes.max[:, :, : dim // 2]
        s_heads = Multiboxes(s_heads_min, s_heads_max)
        s_tails_min = s_multiboxes.min[:, :, dim // 2 :]
        s_tails_max = s_multiboxes.max[:, :, dim // 2 :]
        s_tails = Multiboxes(s_tails_min, s_tails_max)

        return (
            self.inclusion_loss(r1_heads, s_heads)
            + self.inclusion_loss(r2_tails, s_tails)
        ) / 2

    def get_data_batch(self, train_data, key):
        if len(train_data[key]) <= self.batch_size:
            return train_data[key].to(self.device)
        else:
            rand_index = np.random.choice(len(train_data[key]), size=self.batch_size)
            return train_data[key][rand_index].to(self.device)

    def get_negative_sample_batch(self, train_data, key):
        rand_index = np.random.choice(len(train_data[f"{key}0"]), size=self.batch_size)
        neg_data = train_data[f"{key}0"][rand_index]
        for i in range(1, self.num_neg):
            neg_data2 = train_data[f"{key}{i}"][rand_index]
            neg_data = torch.cat([neg_data, neg_data2], dim=0)
        return neg_data.to(self.device)

    def forward(self, train_data):
        loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        nfi_methods = {
            "nf1": self.nf1_loss,
            "nf2": self.nf2_loss,
            "nf3": self.nf3_loss,
            "nf4": self.nf4_loss,
            "disjoint": self.nf2_disjoint_loss,
        }
        for nfi, method in nfi_methods.items():
            if len(train_data[nfi]) > 0:
                nfi_data = self.get_data_batch(train_data, nfi)
                nfi_loss = method(nfi_data).square().mean()
                print(f"{nfi} loss: {nfi_loss}")
                if self.wandb:
                    self.wandb.log({f"{nfi}_loss": nfi_loss})
                loss = loss + nfi_loss

        # if self.num_neg > 0:
        #     neg_data = self.get_negative_sample_batch(train_data, "nf3_neg")
        #     neg_loss1, neg_loss2 = self.nf3_neg_loss(neg_data)
        #     loss += (self.neg_dist - neg_loss1).square().mean() + (
        #         self.neg_dist - neg_loss2
        #     ).square().mean()

        if "abox" in train_data:
            abox = train_data["abox"]
            ra_data = self.get_data_batch(abox, "role_assertions")
            assertion_loss = self.role_assertion_loss(ra_data).square().mean()
            print(f"Role assertion loss: {assertion_loss}")
            loss = loss + assertion_loss

            # neg_data = self.get_negative_sample_batch(abox, "role_assertions_neg")
            # neg_loss1, neg_loss2 = self.role_assertion_neg_loss(neg_data)
            # loss += (self.neg_dist - neg_loss1).square().mean() + (
            #     self.neg_dist - neg_loss2
            # ).square().mean()

            ca_data = self.get_data_batch(abox, "concept_assertions")
            concept_assertion_loss = (
                self.concept_assertion_loss(ca_data).square().mean()
            )
            print(f"Concept assertion loss: {concept_assertion_loss}")
            loss = loss + concept_assertion_loss

        if "role_inclusion" in train_data and len(train_data["role_inclusion"]) > 0:
            ri_data = self.get_data_batch(train_data, "role_inclusion")
            role_inclusion_loss = self.role_inclusion_loss(ri_data).square().mean()
            print(f"Role inclusion loss: {role_inclusion_loss}")
            loss = loss + role_inclusion_loss
        if "role_chain" in train_data and len(train_data["role_chain"]) > 0:
            rc_data = self.get_data_batch(train_data, "role_chain")
            role_chain_loss = self.role_chain_loss(rc_data).square().mean()
            print(f"Role chain loss: {role_chain_loss}")

        # class_reg = (
        #     self.reg_factor
        #     * torch.linalg.norm(self.bumps.weight, dim=1).reshape(-1, 1).mean()
        # )
        # if self.num_individuals > 0:
        #     individual_reg = (
        #         self.reg_factor
        #         * torch.linalg.norm(self.individual_bumps.weight, dim=1)
        #         .reshape(-1, 1)
        #         .mean()
        #     )
        # loss += (class_reg + individual_reg) / 2
        # else:
        #     loss += class_reg

        if self.vis_loss:  # only used for plotting nice boxes
            vis_loss = relu(
                0.2 - torch.abs(self.class_embeds.weight[:, self.embedding_dim :])
            )
            loss += vis_loss.mean()

        return loss

    def to_loaded_model(self):
        model = MultiBoxELLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.class_embeds.weight.detach()
        model.relation_embeds = self.relation_embeds.weight.detach()
        if self.num_individuals > 0:
            model.individual_embeds = self.individual_embeds.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = "_best" if best else ""
        np.save(
            f"{folder}/class_embeds{suffix}.npy",
            self.class_embeds.weight.detach().cpu().numpy(),
        )
        np.save(
            f"{folder}/relation_embeds{suffix}.npy",
            self.relation_embeds.weight.detach().cpu().numpy(),
        )
        if self.num_individuals > 0:
            np.save(
                f"{folder}/individual_embeds{suffix}.npy",
                self.individual_embeds.weight.detach().cpu().numpy(),
            )
