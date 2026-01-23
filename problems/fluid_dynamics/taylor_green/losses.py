import torch
from . import config

def physics_loss(model, x, y, t):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    output = model(x, y, t)
    u = output[:, 0:1]
    v = output[:, 1:2]
    p = output[:, 2:3]

    