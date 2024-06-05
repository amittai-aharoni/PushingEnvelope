import os.path
from abc import ABC, abstractmethod

import numpy as np
import torch

from src.boxes import Boxes
from src.multiboxes import Multiboxes


class LoadedModel(ABC):
    embedding_size: int

    def get_boxes(self, embedding):
        return Boxes(
            embedding[:, : self.embedding_size],
            torch.abs(embedding[:, self.embedding_size :]),
        )

    @abstractmethod
    def is_translational(self):
        pass

    @staticmethod
    def from_name(name, folder, embedding_size, device, best=False):
        model_dict = {
            "boxsqel": BoxSqELLoadedModel,
            "elbe": ElbeLoadedModel,
            "ablation": ElbeLoadedModel,
            "elem": ElbeLoadedModel,
            "EmELpp": ElbeLoadedModel,
            "boxel": BoxELLoadedModel,
            "multiboxel": MultiBoxELLoadedModel,
        }
        return model_dict[name].load(folder, embedding_size, device, best)


class BoxSqELLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    individual_embeds: torch.Tensor
    bumps: torch.Tensor
    individual_bumps: torch.Tensor
    relation_heads: torch.Tensor
    relation_tails: torch.Tensor

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = BoxSqELLoadedModel()
        model.embedding_size = embedding_size
        suffix = "_best" if best else ""
        model.class_embeds = torch.from_numpy(
            np.load(f"{folder}/class_embeds{suffix}.npy")
        ).to(device)
        model.bumps = torch.from_numpy(np.load(f"{folder}/bumps{suffix}.npy")).to(
            device
        )
        model.relation_heads = torch.from_numpy(
            np.load(f"{folder}/rel_heads{suffix}.npy")
        ).to(device)
        model.relation_tails = torch.from_numpy(
            np.load(f"{folder}/rel_tails{suffix}.npy")
        ).to(device)
        if os.path.exists(f"{folder}/individual_embeds{suffix}.npy"):
            model.individual_embeds = torch.from_numpy(
                np.load(f"{folder}/individual_embeds{suffix}.npy")
            ).to(device)
            model.individual_bumps = torch.from_numpy(
                np.load(f"{folder}/individual_bumps{suffix}.npy")
            ).to(device)
        return model


class MultiBoxELLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    individual_embeds: torch.Tensor
    relation_embeds: torch.Tensor

    def get_multiboxes(self, embedding: torch.Tensor, is_relation=False) -> Multiboxes:
        embedding_dim = self.embedding_size * 2 if is_relation else self.embedding_size
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
        return multiboxes

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = MultiBoxELLoadedModel()
        model.embedding_size = embedding_size
        suffix = "_best" if best else ""
        model.class_embeds = torch.from_numpy(
            np.load(f"{folder}/class_embeds{suffix}.npy")
        ).to(device)
        model.relation_embeds = torch.from_numpy(
            np.load(f"{folder}/relation_embeds{suffix}.npy")
        ).to(device)
        if os.path.exists(f"{folder}/individual_embeds{suffix}.npy"):
            model.individual_embeds = torch.from_numpy(
                np.load(f"{folder}/individual_embeds{suffix}.npy")
            ).to(device)
        return model


class ElbeLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    relation_embeds: torch.Tensor

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = ElbeLoadedModel()
        model.embedding_size = embedding_size
        suffix = "_best" if best else ""
        model.class_embeds = torch.from_numpy(
            np.load(f"{folder}/class_embeds{suffix}.npy")
        ).to(device)
        model.relation_embeds = torch.from_numpy(
            np.load(f"{folder}/relations{suffix}.npy")
        ).to(device)
        return model


class BoxELLoadedModel(LoadedModel):
    min_embedding: torch.Tensor
    delta_embedding: torch.Tensor
    relation_embedding: torch.Tensor
    scaling_embedding: torch.Tensor

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = BoxELLoadedModel()
        model.embedding_size = embedding_size
        suffix = "_best" if best else ""
        model.min_embedding = torch.from_numpy(
            np.load(f"{folder}/min_embeds{suffix}.npy")
        ).to(device)
        model.delta_embedding = torch.from_numpy(
            np.load(f"{folder}/delta_embeds{suffix}.npy")
        ).to(device)
        model.relation_embedding = torch.from_numpy(
            np.load(f"{folder}/rel_embeds{suffix}.npy")
        ).to(device)
        model.scaling_embedding = torch.from_numpy(
            np.load(f"{folder}/scaling_embeds{suffix}.npy")
        ).to(device)
        return model
