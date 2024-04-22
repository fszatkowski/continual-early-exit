from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 3"
    " --num-workers 0"
    " --gridsearch-tasks -1"
    " --approach podnet"
    " --max-examples-per-class-trn 200"
    " --max-examples-per-class-val 20"
    " --max-examples-per-class-tst 20"
)


def test_podnet_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --pod-fmap-layers conv1 conv2"
    run_main_and_assert(args_line)


def test_podnet_exemplars_early_exit():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --ic-type standard_conv standard_conv"
    args_line += " --ic-layers conv1 conv2"
    args_line += " --input-size 1 28 28"
    run_main_and_assert(args_line)
