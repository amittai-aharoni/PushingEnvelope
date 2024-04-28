from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


class DataLoader(ABC):
    @abstractmethod
    def load_data(
        self, dataset: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int]]:
        pass

    @abstractmethod
    def load_val_data(
        self, dataset: str, classes: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def load_test_data(
        self, dataset: str, classes: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        pass

    @staticmethod
    def from_task(task: str) -> DataLoader:
        """
        Factory method to create a DataLoader based on the task.
        """
        from src.utils.inferences_data_loader import InferencesDataLoader
        from src.utils.old_data_loader import OldDataLoader
        from src.utils.prediction_data_loader import PredictionDataLoader

        task_to_data_loader = {
            "old": OldDataLoader,
            "inferences": InferencesDataLoader,
            "prediction": PredictionDataLoader,
        }
        return task_to_data_loader[task]()
