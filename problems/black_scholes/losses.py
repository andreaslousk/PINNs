"""Loss functions for Black-Scholes PINN"""
import torch
from torch import nn
from . import config


def physics_loss(model, x_f, target_f, device='cpu'):
    """
    Compute Black-Scholes PDE residual loss.
    
    Black-Scholes PDE:
        ∂C/∂t + 0.5*σ²*S²*∂²C/∂S² + (r-q)*S*∂C/∂S - r*C = 0
    
    Args:
        model: PINN model
        x_f: Collocation points [n, 2] where x_f[:, 0] = S, x_f[:, 1] = t
        target_f: Target (zeros) [n, 1]
        device: Device for computation
    
    Returns:
        MSE of PDE residual
    """
    x_f = x_f.to(device)
    x_f.requires_grad = True
    target_f = target_f.to(device)
    
    # Extract S and t
    s_f = x_f[:, 0].reshape(-1, 1)
    t_f = x_f[:, 1].reshape(-1, 1)
    
    # Model prediction
    pred_opt = model(x_f)
    
    # First derivatives
    pred_opt_first_grad = torch.autograd.grad(
        pred_opt,
        x_f,
        create_graph=True,
        grad_outputs=torch.ones_like(pred_opt)
    )[0]
    
    pred_opt_ds = pred_opt_first_grad[:, 0].reshape(-1, 1)  # ∂C/∂S
    pred_opt_dt = pred_opt_first_grad[:, 1].reshape(-1, 1)  # ∂C/∂t
    
    # Second derivative
    pred_opt_dss = torch.autograd.grad(
        pred_opt_ds,
        x_f,
        create_graph=True,
        grad_outputs=torch.ones_like(pred_opt_ds)
    )[0][:, 0].reshape(-1, 1)  # ∂²C/∂S²
    
    # Black-Scholes PDE residual
    a = 0.5 * (config.sigma ** 2)
    b = (config.r - config.q)
    
    pred_f = (pred_opt_dt + 
             a * (s_f ** 2) * pred_opt_dss + 
             b * s_f * pred_opt_ds - 
             config.r * pred_opt)
    
    # MSE loss
    loss_fn = nn.MSELoss()
    return loss_fn(pred_f, target_f)


def boundary_loss(model, x_b, target_b, device='cpu'):
    """
    Compute boundary condition loss (S=0).
    
    Boundary condition: C(0, t) = 0
    
    Args:
        model: PINN model
        x_b: Boundary points [n, 2]
        target_b: Target values (zeros) [n, 1]
        device: Device for computation
    
    Returns:
        MSE of boundary condition
    """
    x_b = x_b.to(device)
    target_b = target_b.to(device)
    
    pred_b = model(x_b)
    
    loss_fn = nn.MSELoss()
    return loss_fn(pred_b, target_b)


def expiration_loss(model, x_exp, target_exp, device='cpu'):
    """
    Compute terminal condition loss (t=T).
    
    Terminal condition: C(S, T) = max(S - K, 0)
    
    Args:
        model: PINN model
        x_exp: Expiration points [n, 2]
        target_exp: Target payoffs [n, 1]
        device: Device for computation
    
    Returns:
        MSE of terminal condition
    """
    x_exp = x_exp.to(device)
    target_exp = target_exp.to(device)
    
    pred_exp = model(x_exp)
    
    loss_fn = nn.MSELoss()
    return loss_fn(pred_exp, target_exp)


def total_loss(model, x_f, target_f, x_b, target_b, x_exp, target_exp, device='cpu'):
    """
    Compute total weighted loss.
    
    Args:
        model: PINN model
        x_f: Collocation points
        target_f: PDE targets (zeros)
        x_b: Boundary points
        target_b: Boundary targets (zeros)
        x_exp: Expiration points
        target_exp: Expiration targets (payoffs)
        device: Device for computation
    
    Returns:
        total_loss, dict with individual losses
    """
    loss_phys = physics_loss(model, x_f, target_f, device)
    loss_bound = boundary_loss(model, x_b, target_b, device)
    loss_exp = expiration_loss(model, x_exp, target_exp, device)
    
    # Weighted combination
    loss = (config.LOSS_WEIGHT_F * loss_phys + 
            config.LOSS_WEIGHT_B * loss_bound + 
            config.LOSS_WEIGHT_EXP * loss_exp)
    
    # Return total and breakdown
    loss_dict = {
        'total': loss.item(),
        'physics': loss_phys.item(),
        'boundary': loss_bound.item(),
        'expiration': loss_exp.item()
    }
    
    return loss, loss_dict
