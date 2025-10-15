from torch import nn


class Model1(nn.Module):
    def __init__(self, neurons):
        super(Model1, self).__init__()

        self.neurons = neurons

        self.layer1 = nn.Linear(2, self.neurons)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Linear(self.neurons, self.neurons)
        self.act2 = nn.ReLU()

        self.layer3 = nn.Linear(self.neurons, self.neurons)
        self.act3 = nn.ReLU()

        self.layer4 = nn.Linear(self.neurons, self.neurons)
        self.act4 = nn.ReLU()

        self.layer5 = nn.Linear(self.neurons, self.neurons)
        self.act5 = nn.ReLU()

        self.layer6 = nn.Linear(self.neurons, self.neurons)
        self.act6 = nn.ReLU()

        self.layer7 = nn.Linear(self.neurons, self.neurons)
        self.act7 = nn.ReLU()

        self.layer8 = nn.Linear(self.neurons, self.neurons)
        self.act8 = nn.ReLU()

        self.layer9 = nn.Linear(self.neurons, self.neurons)
        self.act9 = nn.ReLU()

        self.layer10 = nn.Linear(self.neurons, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.act2(x)

        x = self.layer3(x)
        x = self.act3(x)

        x = self.layer4(x)
        x = self.act4(x)

        x = self.layer5(x)
        x = self.act5(x)

        x = self.layer6(x)
        x = self.act6(x)

        x = self.layer7(x)
        x = self.act7(x)

        x = self.layer8(x)
        x = self.act8(x)

        x = self.layer9(x)
        x = self.act9(x)

        x = self.layer10(x)

        return x