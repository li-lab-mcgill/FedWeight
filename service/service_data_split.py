import numpy as np
import torch

from utils.utils_log import LogUtils
from utils.constants import *


class DataSplitService:

    """ Initialize """

    def __init__(self, device):
        self._device = device

    """ Public methods """

    def split_data(self, x_target, y_target,
                   test_size, val_size):

        if test_size >= 1 or test_size <= 0 or val_size >= 1 or val_size <= 0:
            raise Exception("Size must be in range 0 and 1")

        total_length = len(y_target)
        total_ids = np.random.permutation(total_length)

        train_size = 1 - test_size - val_size
        if train_size <= 0:
            raise Exception(
                "The sum of test and validation size should not exceed 1")

        LogUtils.instance().log_info("Target train size: {}, val size: {}, test size: {}".format(
            train_size, val_size, test_size))

        test_ids, val_ids, train_ids = np.split(total_ids, [int(
            test_size * total_length), int(
            test_size * total_length) + int(val_size * total_length)])

        assert len(train_ids) + len(val_ids) + len(test_ids) == total_length

        x_target_train = x_target[train_ids]
        y_target_train = y_target[train_ids]

        x_target_val = x_target[val_ids]
        y_target_val = y_target[val_ids]

        x_target_test = x_target[test_ids]
        y_target_test = y_target[test_ids]

        x_target_train = torch.Tensor(x_target_train).to(self._device)
        y_target_train = torch.Tensor(y_target_train).to(self._device)

        x_target_val = torch.Tensor(x_target_val).to(self._device)
        y_target_val = torch.Tensor(y_target_val).to(self._device)

        x_target_test = torch.Tensor(x_target_test).to(self._device)
        y_target_test = torch.Tensor(y_target_test).to(self._device)

        LogUtils.instance().log_info("x_target_train: {}, x_target_val: {}, x_target_test: {}".format(
            x_target_train.shape, x_target_val.shape, x_target_test.shape))
        LogUtils.instance().log_info("y_target_train: {}, y_target_val: {}, y_target_test: {}".format(
            y_target_train.shape, y_target_val.shape, y_target_test.shape))

        return (x_target_train, y_target_train), (x_target_val, y_target_val), (x_target_test, y_target_test)
