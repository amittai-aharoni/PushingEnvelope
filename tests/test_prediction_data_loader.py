import os.path

import pytest

from src.config import PREDICTION_DATA_STRUCTURE
from src.utils.prediction_data_loader import PredictionDataLoader


@pytest.fixture(scope="session")
def prediction_data_loader():
    return PredictionDataLoader()


@pytest.mark.parametrize(
    "dataset",
    [
        "GALEN",
        "GO",
        "ANATOMY",
    ],
)
def test_load_arrays(prediction_data_loader, dataset):
    data_keys = PREDICTION_DATA_STRUCTURE.keys()
    path = prediction_data_loader.get_file_dir(dataset=dataset)
    train_path = os.path.join(path, "train")
    arrays = prediction_data_loader.load_arrays(train_path)
    assert set(arrays.keys()) == set(data_keys)
    for key in data_keys:
        assert key in arrays


def test_load_data(prediction_data_loader):
    dataset = "GALEN"
    data, classes, relations = prediction_data_loader.load_data(dataset)
    assert len(data) == len(PREDICTION_DATA_STRUCTURE)
    assert len(classes) > 0
    assert len(relations) > 0
