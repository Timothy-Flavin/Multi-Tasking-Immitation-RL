import torch
import numpy as np


def T(a, device="cpu", dtype=torch.float32, debug=False):
    if debug:
        print("T: ", a)
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a).to(device)
    elif not torch.is_tensor(a):
        return torch.from_numpy(np.array(a), dtype=dtype).to(device)
    elif a.device != device:
        return a.to(device)
    else:
        return a.to(device)


def get_multi_discrete_one_hot(x, discrete_action_dims, debug=False):
    onehot = torch.zeros((x.shape[0], sum(discrete_action_dims)), device=x.device)
    start = 0
    for i, dim in enumerate(discrete_action_dims):
        onehot[torch.arange(x.shape[0]), x[:, i].long() + start] = 1
        start += dim
    if debug:
        print(f"get_multi_discrete_one_hot: {x}, {discrete_action_dims}, {onehot}")
        # input()
    return onehot


def minmaxnorm(data, mins, maxes):
    data_0_to_1 = (data - mins) / (maxes - mins)
    return data_0_to_1 * 2 - 1


def normgrad(parameters, grad_clip=0.5):
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
