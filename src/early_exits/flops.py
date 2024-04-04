import logging

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from fvcore.nn.jit_handles import elementwise_flop_counter

OP_HANDLERS = {
    # TODO copied from effbench, might require further investigation
    "aten::add": elementwise_flop_counter(0, 1),
    "aten::add_": elementwise_flop_counter(0, 1),
    "aten::radd": elementwise_flop_counter(0, 1),
    "aten::sub": elementwise_flop_counter(0, 1),
    "aten::sub_": elementwise_flop_counter(0, 1),
    "aten::rsub": elementwise_flop_counter(0, 1),
    "aten::mul": elementwise_flop_counter(0, 1),
    "aten::mul_": elementwise_flop_counter(0, 1),
    "aten::rmul": elementwise_flop_counter(0, 1),
    "aten::div": elementwise_flop_counter(0, 1),
    "aten::div_": elementwise_flop_counter(0, 1),
    "aten::rdiv": elementwise_flop_counter(0, 1),
    "aten::exp": elementwise_flop_counter(0, 1),
    "aten::cumsum": elementwise_flop_counter(0, 1),
    "aten::ne": elementwise_flop_counter(0, 1),
    "aten::lt": elementwise_flop_counter(0, 1),
    "aten::gelu": elementwise_flop_counter(0, 1),
    "aten::abs": elementwise_flop_counter(0, 1),
    "aten::silu_": elementwise_flop_counter(0, 1),
    "aten::dropout_": elementwise_flop_counter(0, 1),
    "aten::sigmoid": elementwise_flop_counter(0, 1),
    "aten::tanh": elementwise_flop_counter(0, 1),
    "aten::softmax": elementwise_flop_counter(0, 2),
    "aten::log_softmax": elementwise_flop_counter(0, 2),
    "aten::argmax": elementwise_flop_counter(0, 1),
    "aten::one_hot": elementwise_flop_counter(0, 1),
    "aten::flatten": elementwise_flop_counter(0, 0),
    "aten::unflatten": elementwise_flop_counter(0, 0),
    "aten::mean": elementwise_flop_counter(1, 0),
    "aten::sum": elementwise_flop_counter(1, 0),
    "aten::topk": elementwise_flop_counter(1, 1),
    "aten::scatter": elementwise_flop_counter(1, 1),
    "aten::scatter_": elementwise_flop_counter(1, 1),
    "aten::gather": elementwise_flop_counter(1, 1),
    "aten::gather_": elementwise_flop_counter(1, 1),
    "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
}


def analyze_flops(model: torch.nn.Module, input):
    model_costs = FlopCountAnalysis(model, input).set_op_handle(**OP_HANDLERS)
    total_flops = model_costs.total()
    unsupported = model_costs.unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f"Unsupported op: {k} (occurrences: {v})")
    uncalled = model_costs.uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f"Uncalled module: {m}")

    param_count = parameter_count(model)

    return total_flops, param_count
