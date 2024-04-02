import torch
from torch import nn


def create_ic(ic_type, ic_input_size, num_outputs):
    if ic_type == "standard_conv":
        return StandardConvHead(ic_input_size, num_outputs)
    elif ic_type == "standard_fc":
        return nn.Linear(ic_input_size, num_outputs)
    else:
        raise NotImplementedError()


class StandardConvHead(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        self.classifier = nn.Linear(input_features // 4, num_classes)

    def forward(self, x):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        return self.classifier(pool_output.view(pool_output.size(0), -1))


def get_sdn_weights(current_epoch, total_epochs):
    starting_weights = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    current_weights = [
        starting_weight + (current_epoch / (total_epochs - 1)) * (1 - starting_weight)
        for starting_weight in starting_weights
    ]
    return current_weights
