import torch
from joblib import Memory

cachedir = "cache"
memory = Memory(cachedir, verbose=0)
printed_device = False


def get_device():
    global printed_device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    if not printed_device:
        print(f"Using device: {device_name}")
        printed_device = True
    return device
