import json
import os
from typing import Dict

import numpy as np
import torch

from src.config import DATA_PATH, ROOT_PATH
from src.utils.data_loader import DataLoader
from src.utils.utils import get_device

device = get_device()


class PredictionDataLoader(DataLoader):
    @staticmethod
    def get_file_dir(dataset: str) -> str:
        path = os.path.join(ROOT_PATH, DATA_PATH, dataset, "prediction")
        return path

    @staticmethod
    def load_arrays(path) -> Dict[str, torch.Tensor]:
        """
        A class used to load prediction data for a given dataset.

        Args:
            path:
            path to train, test or val folder
        Returns:

        """
        d = {}
        for file in os.listdir(path):
            arr = np.load(f"{path}/{file}")
            file = file.replace(".npy", "")
            d[file] = torch.from_numpy(arr).long()
        return d

    def load_val_data(self, dataset, classes):
        return self.load_valid_or_test_data(dataset, "val")

    def load_test_data(self, dataset, classes):
        return self.load_valid_or_test_data(dataset, "test")

    def load_valid_or_test_data(
        self, dataset: str, folder: str
    ) -> Dict[str, torch.Tensor]:
        path = f"{self.get_file_dir(dataset)}/{folder}"
        return self.load_arrays(path)

    def load_data(self, dataset: str):
        folder = self.get_file_dir(dataset)
        data = self.load_arrays(f"{folder}/train")
        with open(f"{folder}/classes.json", "r") as f:
            classes = json.load(f)
        with open(f"{folder}/relations.json", "r") as f:
            relations = json.load(f)
        return data, classes, relations
