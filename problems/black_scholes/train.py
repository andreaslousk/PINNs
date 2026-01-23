"""Training script for Black-Scholes PINN"""
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from . import config
from .model import BlackScholesPinn
from .exact_solution import EuropeanCallOption
from .data import TrainingDataLoader
from .losses import physics_loss, boundary_loss, expiration_loss


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """Get appropriate device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train():
    """Main training function"""
    
    # Set seed
    set_seed(config.SEED)
    print(f"Random seed set to: {config.SEED}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directories
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR + '/checkpoints').mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR + '/logs').mkdir(parents=True, exist_ok=True)
    
    # Initialize option
    print("\nInitializing European Call Option...")
    option = EuropeanCallOption()
    
    # Generate training data
    print("Generating training data...")
    data_loader = TrainingDataLoader(
        option,
        config.LOWER_DELTA_BOUND,
        config.UPPER_DELTA_BOUND,
        config.NUM_COLLOCATION_POINTS,
        config.NUM_EXPIRATION_POINTS,
        config.NUM_BOUNDARY_POINTS
    )
    
    xF, pricesF, targetF, gammaWeightsF = data_loader.get_collocation_training_data()
    xB, targetB = data_loader.get_boundary_training_data()
    xExp, targetExp = data_loader.get_expiration_training_data()
    
    print(f"Collocation points: {len(xF)}")
    print(f"Boundary points: {len(xB)}")
    print(f"Expiration points: {len(xExp)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = BlackScholesPinn().to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_GAMMA
        )
    
    # Create data loaders
    datasetF = TensorDataset(xF, targetF)
    datasetB = TensorDataset(xB, targetB)
    datasetExp = TensorDataset(xExp, targetExp)
    
    loaderF = DataLoader(datasetF, batch_size=config.BATCH_SIZE, shuffle=True)
    loaderB = DataLoader(datasetB, batch_size=config.BATCH_SIZE, shuffle=True)
    loaderExp = DataLoader(datasetExp, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Training history
    history = {
        'loss_f': [],
        'loss_b': [],
        'loss_exp': [],
        'total': []
    }
    
    best_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    model.train()
    
    for epoch in tqdm(range(config.NUM_EPOCHS), desc="Training"):
        epoch_loss_f = 0.0
        epoch_loss_b = 0.0
        epoch_loss_exp = 0.0
        
        # Create iterators
        iter_f = iter(loaderF)
        iter_b = iter(loaderB)
        iter_exp = iter(loaderExp)
        
        max_batches = max(len(loaderF), len(loaderB), len(loaderExp))
        
        for _ in range(max_batches):
            # Get batches (cycle if necessary)
            try:
                x_f_batch, target_f_batch = next(iter_f)
            except StopIteration:
                iter_f = iter(loaderF)
                x_f_batch, target_f_batch = next(iter_f)
            
            try:
                x_b_batch, target_b_batch = next(iter_b)
            except StopIteration:
                iter_b = iter(loaderB)
                x_b_batch, target_b_batch = next(iter_b)
            
            try:
                x_exp_batch, target_exp_batch = next(iter_exp)
            except StopIteration:
                iter_exp = iter(loaderExp)
                x_exp_batch, target_exp_batch = next(iter_exp)
            
            # Compute losses
            loss_f = physics_loss(model, x_f_batch, target_f_batch, device)
            loss_b = boundary_loss(model, x_b_batch, target_b_batch, device)
            loss_exp = expiration_loss(model, x_exp_batch, target_exp_batch, device)
            
            # Total weighted loss
            loss_total = (config.LOSS_WEIGHT_F * loss_f + 
                         config.LOSS_WEIGHT_B * loss_b + 
                         config.LOSS_WEIGHT_EXP * loss_exp)
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient clipping
            if config.GRAD_CLIP_MAX_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.GRAD_CLIP_MAX_NORM
                )
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss_f += loss_f.item()
            epoch_loss_b += loss_b.item()
            epoch_loss_exp += loss_exp.item()
        
        # Average losses
        avg_loss_f = epoch_loss_f / max_batches
        avg_loss_b = epoch_loss_b / max_batches
        avg_loss_exp = epoch_loss_exp / max_batches
        avg_total = (config.LOSS_WEIGHT_F * avg_loss_f + 
                    config.LOSS_WEIGHT_B * avg_loss_b + 
                    config.LOSS_WEIGHT_EXP * avg_loss_exp)
        
        # Store history
        history['loss_f'].append(avg_loss_f)
        history['loss_b'].append(avg_loss_b)
        history['loss_exp'].append(avg_loss_exp)
        history['total'].append(avg_total)
        
        # Save best model
        if config.SAVE_BEST_MODEL and avg_total < best_loss:
            best_loss = avg_total
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        
        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        if epoch % config.SAVE_FREQ == 0:
            print(f'\nEpoch {epoch}:')
            print(f'  Loss Physics    : {avg_loss_f:.6f}')
            print(f'  Loss Boundary   : {avg_loss_b:.6f}')
            print(f'  Loss Expiration : {avg_loss_exp:.6f}')
            print(f'  Total Loss      : {avg_total:.6f}')
            if config.SAVE_BEST_MODEL and epoch == best_epoch:
                print(f'  💾 New best model saved!')
    
    # Save final model
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Final model saved to: {config.FINAL_MODEL_PATH}")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH} (epoch {best_epoch})")
    
    # Save losses
    np.save(config.OUTPUT_DIR + '/logs/loss_f.npy', np.array(history['loss_f']))
    np.save(config.OUTPUT_DIR + '/logs/loss_b.npy', np.array(history['loss_b']))
    np.save(config.OUTPUT_DIR + '/logs/loss_exp.npy', np.array(history['loss_exp']))
    print(f"Training history saved to: {config.OUTPUT_DIR}/logs/")
    print("="*80)


if __name__ == "__main__":
    train()
