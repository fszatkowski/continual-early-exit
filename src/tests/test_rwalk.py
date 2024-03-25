from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach r_walk"


def test_rwalk_without_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 0"
    run_main_and_assert(args_line)


def test_rwalk_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)
