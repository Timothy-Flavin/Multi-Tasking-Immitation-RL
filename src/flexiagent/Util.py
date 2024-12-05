import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def T(a, device="cpu", dtype=torch.float32):
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a).to(device)
    elif not torch.is_tensor(a):
        return torch.from_numpy(np.array(a), dtype=dtype).to(device)
