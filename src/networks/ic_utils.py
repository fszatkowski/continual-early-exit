import math
from typing import Tuple

import torch
from torch import nn


def create_ic(ic_type: str, ic_input_size: Tuple[int, ...], num_outputs: int):
    if ic_type == "standard_conv":
        return StandardConvHead(ic_input_size, num_outputs)
    elif ic_type == "adaptive_standard_conv":
        return AdaptiveStandardConvHead(ic_input_size, num_outputs)
    elif ic_type == "basic_conv":
        return BasicConvHead(ic_input_size, num_outputs)
    elif ic_type == "standard_fc":
        num_ic_features = math.prod(ic_input_size)
        return nn.Linear(num_ic_features, num_outputs)
    else:
        raise NotImplementedError()


class BasicConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        num_input_features = math.prod(input_features)
        self.classifier = nn.Linear(num_input_features // 4, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.maxpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class AdaptiveStandardConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        _, c, h, w = input_features
        h_new = math.ceil(h / 2)
        w_new = math.ceil(w / 2)
        self.maxpool = nn.AdaptiveMaxPool2d((h_new, w_new))
        self.avgpool = nn.AdaptiveAvgPool2d((h_new, w_new))
        self.alpha = nn.Parameter(torch.rand(1))
        self.classifier = nn.Linear(c * h_new * w_new, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class StandardConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        num_input_features = math.prod(input_features)
        self.classifier = nn.Linear(num_input_features // 4, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


def get_sdn_weights(current_epoch, total_epochs, n_ics):
    if n_ics == 6:
        final_weights = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    elif n_ics == 7:
        final_weights = [0.2, 0.3, 0.4, 0.55, 0.65, 0.75, 0.9]
    else:
        raise NotImplementedError()

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


class RegisterForwardHook:
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        self.output = output


def register_intermediate_output_hooks(model, layers):
    hooks = []
    modules = [(name, module) for name, module in model.named_modules()]
    for layer_name in layers:
        module_found = False
        for module_name, module in modules:
            if module_name == layer_name:
                hook = RegisterForwardHook()
                module.register_forward_hook(hook)
                hooks.append(hook)
                print(f"Attaching IC to the layer {layer_name}...")
                module_found = True
                break
        if not module_found:
            raise ValueError(f"Could not find module {layer_name} to attach the IC.")
    return hooks
