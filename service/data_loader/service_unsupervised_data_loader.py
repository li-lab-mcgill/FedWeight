from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService

from utils.utils_log import LogUtils
from utils.constants import *

from typing import Dict, Tuple
import pandas as pd
import numpy as np


class UnsupervisedDataLoaderService(AbstractDataLoaderService):

    def load_data(self, **kwargs) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:

        result = dict()

        dataset_path = kwargs.get("dataset_path")
        hospital_id_col = kwargs.get("hospital_id_col")
        dedicated_hospital_ids = kwargs.get("dedicated_hospital_ids")
        task = kwargs.get("task")
        run_with_fl = kwargs.get("run_with_fl")

        assert dataset_path is not None
        assert dedicated_hospital_ids is not None
        assert hospital_id_col is not None
        assert task is not None

        icu_data = pd.read_csv(dataset_path)
        LogUtils.instance().log_info(
            "Dataset shape: {}".format(icu_data.shape))

        # Select dedicated hospitals
        for idx, hospital_id in enumerate(dedicated_hospital_ids):
            hospital_data = icu_data[icu_data["hospitalid"] == hospital_id]

            x = hospital_data.iloc[:, 1:-1].to_numpy()
            if run_with_fl:
                cls_x = x
            else:
                one_hot_hospital_ids = np.zeros((len(x), len(dedicated_hospital_ids)))
                one_hot_hospital_ids[:, idx] = 1.0
                hospital_x = np.concatenate((one_hot_hospital_ids, x), axis=1)
                cls_x = hospital_x

            density_x = x
            cls_y = np.zeros((len(x), 1))

            result[hospital_id] = (hospital_id, cls_x, cls_y, density_x)
            LogUtils.instance().log_info('Hospital: {}, Total patient count: {}'.format(
                int(hospital_id), len(hospital_data)))

        return result

    def is_eligible(self, task: str) -> bool:
        return task == UNSUPERVISED
