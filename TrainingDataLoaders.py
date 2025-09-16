import torch

class TrainingDataLoader:
    def __init__(self, option, lowerDeltaBound, upperDeltaBound, numCollPointsTrain, numExpPointsTrain, numBoundaryPointsTrain):
        self.option = option
        
        self.lowerDeltaBound = lowerDeltaBound
        self.upperDeltaBound = upperDeltaBound

        self.numCollPointsTrain = numCollPointsTrain
        self.numExpPointsTrain = numExpPointsTrain
        self.numBoundaryPointsTrain = numBoundaryPointsTrain
        
        collPoints, callPrices, gammaWeights = self.option.filterDataByDelta(self.lowerDeltaBound, self.upperDeltaBound)
        numCollPoints = callPrices.size()[0]

        assert self.numCollPointsTrain <= numCollPoints, f"Cannot select {self.numCollPointsTrain} collocation points from a sample size of {numCollPoints}."
        
        collShuffle = torch.randperm(numCollPoints)[ : self.numCollPointsTrain]
        self.collPointsTrain = collPoints[collShuffle]
        self.callPricesCompare = callPrices[collShuffle]
        self.collTarget = torch.zeros(self.numCollPointsTrain, 1)
        self.collGammaWeights = gammaWeights[collShuffle]
        
        gridPoints = option.gridPoints
        
        expPoints, callPricesExp = gridPoints[-1], option.C[-1, : ]
        numExpPoints = expPoints.size()[0]

        assert self.numExpPointsTrain <= numExpPoints, f"Cannot select {self.numExpPointsTrain} expiration points from a sample size of {numCollPoints}."

        expShuffle = torch.randperm(numExpPoints)[ : self.numExpPointsTrain]
        self.expPointsTrain = expPoints[expShuffle]
        self.expTarget = callPricesExp[expShuffle]

        boundaryPoints, callPricesBoundary = gridPoints[ : , 0], option.C[ : , 0]
        numBoundaryPoints = boundaryPoints.size()[0]
        
        assert self.numBoundaryPointsTrain <= numBoundaryPoints, f"Cannot select {self.numBoundaryPointsTrain} expiration points from a sample size of {numBoundaryPoints}."

        boundaryShuffle = torch.randperm(numBoundaryPoints)[ : self.numBoundaryPointsTrain]
        self.boundaryPointsTrain = boundaryPoints[boundaryShuffle]
        self.boundaryTarget = callPricesBoundary[boundaryShuffle]

    def getCollocationTrainingData():
        return self.collPointsTrain, self.callPricesCompare, self.collTarget, self.collGammaWeights

    def getExpirationTrainingData():
        return self.expPointsTrain, self.expTarget

    def getBoundaryTrainingData():
        return self.boundaryPointsTrain, self.boundaryTarget