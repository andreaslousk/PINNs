import torch
import random

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from EuropeanCallOption import EuropeanCallOption
from TrainingDataLoaders import TrainingDataLoader
from ModelArchitecture import Model1
from ModelImplementation import ModelImplementation


def setSeed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)

    random.seed(seed)


def main():
    # Set random seed for reproducibility
    SEED = 42
    setSeed(SEED)

    print(f"Random seed set to: {SEED}")

    # Hyperparameters
    EPOCHS = 531
    LEARNING_RATE = 0.001

    # Black-Scholes parameters
    T = 1.0
    K = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    M = 1000
    N = 1000

    option = EuropeanCallOption(K, T, r, sigma, q, M, N)

    # Data filtering and sampling
    lowerDeltaBound = 0.35
    upperDeltaBound = 0.65
    numCollPointsTrain = 15000
    numExpPointsTrain = 300
    numBoundaryPointsTrain = 300

    trainingDataLoader = TrainingDataLoader(
        option, lowerDeltaBound, upperDeltaBound, numCollPointsTrain, numExpPointsTrain, numBoundaryPointsTrain)

    model = Model1(neurons=50)

    xF, pricesF, targetF, gammaWeightsF = trainingDataLoader.getCollocationTrainingData()
    xB, targetB = trainingDataLoader.getBoundaryTrainingData()
    xExp, targetExp = trainingDataLoader.getExpirationTrainingData()

    BATCH_SIZE = 64

    # Optimizer and learning rate scheduler (optional)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    trainer = ModelImplementation(model, xF, targetF, xB, targetB, xExp, targetExp,
                                  optimizer, EPOCHS, T, K, r, q, sigma, batchSize=BATCH_SIZE, learningRateScheduler=scheduler)

    trainer.initializeModel()

    print("Starting training...")
    trainer.train()

    # Save model
    modelPath = 'trainedPinnModel.pth'
    torch.save(model.state_dict(), modelPath)
    print(f"\nModel saved to {modelPath}")

    # Save losses
    np.save('lossB.npy', np.array(trainer.lossB))
    np.save('lossExp.npy', np.array(trainer.lossExp))
    np.save('lossF.npy', np.array(trainer.lossF))
    print("Losses saved as .npy files")


if __name__ == "__main__":
    main()