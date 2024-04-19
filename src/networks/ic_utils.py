import torch
from torch import nn


def create_ic(ic_type, ic_input_size, num_outputs):
    if ic_type == "standard_conv":
        return StandardConvHead(ic_input_size, num_outputs)
    elif ic_type == "basic_conv":
        return BasicConvHead(ic_input_size, num_outputs)
    elif ic_type == "standard_fc":
        return nn.Linear(ic_input_size, num_outputs)
    else:
        raise NotImplementedError()


class BasicConvHead(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Linear(input_features // 4, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.maxpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class StandardConvHead(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        self.classifier = nn.Linear(input_features // 4, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


def get_sdn_weights(current_epoch, total_epochs):
    final_weights = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    start_val = 0.01
    current_weights = [
        start_val + (current_epoch / (total_epochs - 1)) * (final_weight - start_val)
        for final_weight in final_weights
    ]
    current_weights += [1.0]
    return current_weights


def get_alt_sdn_weights(current_epoch, total_epochs):
    starting_weights = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    current_weights = [
        starting_weight + (current_epoch / (total_epochs - 1)) * (1 - starting_weight)
        for starting_weight in starting_weights
    ]
    return current_weights
