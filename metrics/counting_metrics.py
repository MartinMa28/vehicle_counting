import torch

def mae(et_dm, gt_dm):
    return torch.abs(et_dm.sum() - gt_dm.sum())

def mse(et_dm, gt_dm):
    return (et_dm.sum() - get_dm.sum()) * (et_dm.sum() - get_dm.sum())