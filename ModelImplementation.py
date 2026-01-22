import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class ModelImplementation:
    def __init__(self, model, xF, targetF, xB, targetB, xExp, targetExp,
                 optimizer, epochs, T, K, r, q, sigma, batchSize=32,
                 lossWeightF=100, lossWeightB=1, lossWeightExp=1,
                 gradClipMaxNorm=1.0, learningRateScheduler=None, saveBestModel=True, modelSavePath='bestModel.pth'):
        self.saveBestModel = saveBestModel
        self.modelSavePath = modelSavePath
        self.bestLoss = float('inf')
        self.bestEpoch = 0
        
        self.model = model

        self.T = T
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma

        # Store batch size
        self.batchSize = batchSize

        # Loss weights
        self.lossWeightF = lossWeightF
        self.lossWeightB = lossWeightB
        self.lossWeightExp = lossWeightExp

        # Gradient clipping
        self.gradClipMaxNorm = gradClipMaxNorm

        # Learning rate scheduler
        self.scheduler = learningRateScheduler

        # Create datasets
        self.datasetF = TensorDataset(xF, targetF)
        self.datasetB = TensorDataset(xB, targetB)
        self.datasetExp = TensorDataset(xExp, targetExp)

        self.nF = len(xF)
        self.nB = len(xB)
        self.nExp = len(xExp)

        self.optimizer = optimizer
        self.epochs = epochs

        self.lossF = []
        self.lossB = []
        self.lossExp = []

        # Device
        self.device = None

    def getDevice(self):
        """Get the appropriate device for computation"""
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("MPS is available. Using MPS for computations.")
            else:
                self.device = torch.device("cpu")
                print("MPS is not available. Using CPU.")
        return self.device

    def initializeModel(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant_(module.bias, 0)

    def train(self):
        # Setup device
        device = self.getDevice()
        self.model = self.model.to(device)
        self.model.train()  # Set model to training mode

        # Create DataLoaders with shuffling
        loaderF = DataLoader(self.datasetF, batch_size=self.batchSize,
                             shuffle=True, drop_last=False)
        loaderB = DataLoader(self.datasetB, batch_size=self.batchSize,
                             shuffle=True, drop_last=False)
        loaderExp = DataLoader(self.datasetExp, batch_size=self.batchSize,
                               shuffle=True, drop_last=False)

        lossFn = nn.MSELoss()

        a = 0.5 * (self.sigma ** 2)
        b = (self.r - self.q)

        # Use tqdm for progress bar
        for epoch in tqdm(range(self.epochs), desc="Training"):
            epochLossB = 0.0
            epochLossExp = 0.0
            epochLossF = 0.0

            # Create iterators for each loader
            iterF = iter(loaderF)
            iterB = iter(loaderB)
            iterExp = iter(loaderExp)

            # Determine the maximum number of batches
            numBatchesF = len(loaderF)
            numBatchesB = len(loaderB)
            numBatchesExp = len(loaderExp)
            maxBatches = max(numBatchesF, numBatchesB, numBatchesExp)

            for _ in range(maxBatches):
                # Get batch from each loader, cycling if necessary
                try:
                    xFBatch, targetFBatch = next(iterF)
                except StopIteration:
                    iterF = iter(loaderF)
                    xFBatch, targetFBatch = next(iterF)

                try:
                    xBBatch, targetBBatch = next(iterB)
                except StopIteration:
                    iterB = iter(loaderB)
                    xBBatch, targetBBatch = next(iterB)

                try:
                    xExpBatch, targetExpBatch = next(iterExp)
                except StopIteration:
                    iterExp = iter(loaderExp)
                    xExpBatch, targetExpBatch = next(iterExp)

                # Move batches to device
                xFBatch = xFBatch.to(device)
                xFBatch.requires_grad = True
                targetFBatch = targetFBatch.to(device)

                xBBatch = xBBatch.to(device)
                targetBBatch = targetBBatch.to(device)

                xExpBatch = xExpBatch.to(device)
                targetExpBatch = targetExpBatch.to(device)

                # Boundary condition loss (S=0)
                predB = self.model(xBBatch)
                mseB = lossFn(predB, targetBBatch)

                # Expiration condition loss
                predExp = self.model(xExpBatch)
                mseExp = lossFn(predExp, targetExpBatch)

                # Physics-informed loss (PDE collocation points)
                sF = xFBatch[:, 0].reshape(-1, 1)
                tF = xFBatch[:, 1].reshape(-1, 1)

                predOpt = self.model(xFBatch)
                predOptFirstGrad = torch.autograd.grad(
                    predOpt,
                    xFBatch,
                    create_graph=True,
                    grad_outputs=torch.ones_like(predOpt)
                )[0]

                predOptDS = predOptFirstGrad[:, 0].reshape(-1, 1)
                predOptDt = predOptFirstGrad[:, 1].reshape(-1, 1)
                predOptDSS = torch.autograd.grad(
                    predOptDS,
                    xFBatch,
                    create_graph=True,
                    grad_outputs=torch.ones_like(predOptDS)
                )[0][:, 0].reshape(-1, 1)

                predF = (predOptDt
                         + a * (sF ** 2) * predOptDSS
                         + b * sF * predOptDS
                         - self.r * predOpt)

                mseF = lossFn(predF, targetFBatch)

                # Total loss with configurable weights
                mseTotal = (self.lossWeightB * mseB +
                            self.lossWeightExp * mseExp +
                            self.lossWeightF * mseF)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                mseTotal.backward()

                # Gradient clipping
                if self.gradClipMaxNorm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.gradClipMaxNorm)

                self.optimizer.step()

                # Accumulate losses
                epochLossB += mseB.item()
                epochLossExp += mseExp.item()
                epochLossF += mseF.item()

            # Average losses over batches
            avgLossB = epochLossB / maxBatches
            avgLossExp = epochLossExp / maxBatches
            avgLossF = epochLossF / maxBatches

            # Calculate total loss for this epoch
            avgTotalLoss = (self.lossWeightB * avgLossB + 
                           self.lossWeightExp * avgLossExp + 
                           self.lossWeightF * avgLossF)

            # Save best model
            if self.saveBestModel and avgTotalLoss < self.bestLoss:
                self.bestLoss = avgTotalLoss
                self.bestEpoch = epoch
                torch.save(self.model.state_dict(), self.modelSavePath)
                print(f'\nEpoch {epoch}:')
                print(f'  Loss S=0        : {avgLossB:.4f}')
                print(f'  Loss Expiration : {avgLossExp:.4f}')
                print(f'  Loss Collocation: {avgLossF:.4f}')
                print(f"  💾 New best model saved! (Total loss: {avgTotalLoss:.6f})")
            
            # Store losses
            self.lossB.append(avgLossB)
            self.lossExp.append(avgLossExp)
            self.lossF.append(avgLossF)

            # Step the learning rate scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

            # Print progress
            if (epoch % 10) == 0:
                print(f'\nEpoch {epoch}:')
                print(f'  Loss S=0        : {avgLossB:.4f}')
                print(f'  Loss Expiration : {avgLossExp:.4f}')
                print(f'  Loss Collocation: {avgLossF:.4f}')

        # Print final epoch losses
        print('\n' + '='*50)
        print('Training Complete!')
        print(f'Final Epoch {self.epochs - 1}:')
        print(f'  Loss S=0        : {self.lossB[-1]:.4f}')
        print(f'  Loss Expiration : {self.lossExp[-1]:.4f}')
        print(f'  Loss Collocation: {self.lossF[-1]:.4f}')
        print('='*50)

    def predict(self, x):
        device = self.getDevice()
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            x = x.to(device)
            return self.model(x)