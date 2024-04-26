import os
from typing import Dict, Optional

DATA_PATH = "data/"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET = "GALEN"
DEFAULT_EPOCHS = 5000
DEFAULT_NUM_NEG = 1
EMBEDDING_DIM = 200

PREDICTION_DATA_STRUCTURE = {
    "top": 1,
    "role_chain": 3,
    "class_ids": 1,
    "role_inclusion": 2,
    "nf4": 3,
    "disjoint": 2,
    "nf1": 2,
    "nf2": 3,
    "nf3": 3,
}


class PARAMS:
    def __init__(
        self,
        dataset: str,
        task: str,
        lr: float,
        margin: float,
        neg_dist: float,
        num_neg: int,
        reg_factor: float,
        epochs: int = DEFAULT_EPOCHS,
        lr_schedule: Optional[int] = None,
    ):
        self.dataset = dataset
        self.task = task
        self.lr = lr
        self.margin = margin
        self.neg_dist = neg_dist
        self.num_neg = num_neg
        self.reg_factor = reg_factor
        self.epochs = epochs
        self.lr_schedule = lr_schedule

    def to_wb_config(self) -> Dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "lr": self.lr,
            "margin": self.margin,
            "neg_dist": self.neg_dist,
            "num_neg": self.num_neg,
            "reg_factor": self.reg_factor,
            "epochs": self.epochs,
            "lr_schedule": self.lr_schedule,
        }


class DATASET_TO_TASKS:
    def __init__(self, tasks: Dict[str, PARAMS]):
        self.tasks = tasks


GALEN = DATASET_TO_TASKS(
    tasks={
        "prediction": PARAMS(
            dataset="GALEN",
            task="prediction",
            lr=0.01,
            lr_schedule=None,
            margin=0.15,
            neg_dist=5,
            num_neg=1,
            reg_factor=0.4,
        ),
        "inferences": PARAMS(
            dataset="GALEN",
            task="inferences",
            lr=0.005,
            lr_schedule=2000,
            margin=0.05,
            neg_dist=1,
            num_neg=2,
            reg_factor=0,
        ),
    }
)

GO = DATASET_TO_TASKS(
    tasks={
        "prediction": PARAMS(
            dataset="GO",
            task="prediction",
            lr=0.01,
            lr_schedule=None,
            margin=0.15,
            neg_dist=5.5,
            num_neg=5,
            reg_factor=0.5,
        ),
        "inferences": PARAMS(
            dataset="GO",
            task="inferences",
            lr=0.005,
            lr_schedule=None,
            margin=0.05,
            neg_dist=3,
            num_neg=3,
            reg_factor=0.05,
        ),
    }
)

ANATOMY = DATASET_TO_TASKS(
    tasks={
        "prediction": PARAMS(
            dataset="ANATOMY",
            task="prediction",
            lr=0.01,
            lr_schedule=None,
            margin=0.05,
            neg_dist=5.5,
            num_neg=3,
            reg_factor=0.3,
        ),
        "inferences": PARAMS(
            dataset="ANATOMY",
            task="inferences",
            lr=0.001,
            lr_schedule=None,
            margin=0.05,
            neg_dist=2,
            num_neg=2,
            reg_factor=0.05,
        ),
    }
)

PPI = DATASET_TO_TASKS(
    tasks={
        "yeast": PARAMS(
            dataset="yeast",
            task="yeast",
            lr=0.01,
            lr_schedule=2000,
            margin=0.02,
            neg_dist=2.5,
            num_neg=4,
            reg_factor=0.2,
        ),
        "human": PARAMS(
            dataset="human",
            task="human",
            lr=0.01,
            lr_schedule=None,
            margin=0.005,
            neg_dist=3.5,
            num_neg=5,
            reg_factor=0.3,
        ),
    }
)

DATASETS = {
    "GALEN": GALEN,
    "GO": GO,
    "ANATOMY": ANATOMY,
    "PPI": PPI,
}
