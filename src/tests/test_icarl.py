from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 3"
    " --num-workers 0"
    " --approach icarl"
)


def test_icarl_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --lamb 1"
    run_main_and_assert(args_line)


def test_icarl_exemplars_without_lamb():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --lamb 0"
    run_main_and_assert(args_line)


def test_icarl_early_exits():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --lamb 1"
    args_line += " --ic-type standard_conv standard_conv standard_fc"
    args_line += " --ic-layers conv1 conv2 fc1"
    args_line += " --input-size 1 28 28"
    run_main_and_assert(args_line)
