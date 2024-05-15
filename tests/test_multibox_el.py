import pytest

from src.model.MultiBoxEL import MultiBoxEL
from src.utils.data_loader import DataLoader


@pytest.fixture(scope="session")
def GALEN_data():
    data_loader = DataLoader.from_task("prediction")
    train_data, classes, relations = data_loader.load_data("GALEN")
    return train_data, classes, relations


@pytest.fixture(scope="session")
def model(GALEN_data):
    return MultiBoxEL(
        device="cpu",
        embedding_dim=6,
        num_classes=len(GALEN_data[1]),
        num_boxes_per_class=2,
        num_roles=len(GALEN_data[2]),
    )


def test_forward(model, GALEN_data):
    data = GALEN_data[0]
    model.forward(data)
    assert True
