import torch

from typing import Tuple


def generate_collocation(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    t = torch.rand(n, 1)

    return x, y, t


def generate_boundary_points(n: int) -> dict:
    x_top = torch.rand(n, 1)
    y_top = torch.ones(n, 1)
    t_top = torch.rand(n, 1)

    x_bottom = torch.rand(n, 1)
    y_bottom = torch.zeros(n, 1)
    t_bottom = torch.rand(n, 1)

    x_left = torch.zeros(n, 1)
    y_left = torch.rand(n, 1)
    t_left = torch.rand(n, 1)

    x_right = torch.ones(n, 1)
    y_right = torch.rand(n, 1)
    t_right = torch.rand(n, 1)

    return {
        'top': (x_top, y_top, t_top),
        'bottom': (x_bottom, y_bottom, t_bottom),
        'left': (x_left, y_left, t_left),
        'right': (x_right, y_right, t_right)
    }


def generate_initial_points(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    t = torch.rand(n, 1)

    return x, y, t


def scale_points(points: torch.Tensor, max_val: float) -> torch.Tensor:
    return points * max_val


def generate_all_points(n_collocation: int, n_boundary: int, n_initial: int, device: str = 'cpu') -> dict:
    x_col, y_col, t_col = generate_collocation(n_collocation)

    boundary_points = generate_boundary_points(n_boundary)

    x_init, y_init, t_init = generate_initial_points(n_initial)

    data = {
        'collocation': (x_col.to(device), y_col.to(device), t_col.to(device)),
        'boundary': {
            'top': tuple(p.to(device) for p in boundary_points['top']),
            'bottom': tuple(p.to(device) for p in boundary_points['bottom']),
            'left': tuple(p.to(device) for p in boundary_points['left']),
            'right': tuple(p.to(device) for p in boundary_points['right'])
        },
        'initial': (x_init.to(device), y_init.to(device), t_init.to(device))
    }

    return data
