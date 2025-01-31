from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService

from utils.constants import *

from typing import Dict, Tuple
import pandas as pd
import numpy as np


class SimulateDataLoaderService(AbstractDataLoaderService):

    def load_data(self, **kwargs) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

        result = dict()

        simulate_x_source_path = kwargs.get("simulate_x_source_path")
        simulate_y_source_path = kwargs.get("simulate_y_source_path")
        simulate_x_target_path = kwargs.get("simulate_x_target_path")
        simulate_y_target_path = kwargs.get("simulate_y_target_path")

        assert simulate_x_source_path is not None
        assert simulate_y_source_path is not None
        assert simulate_x_target_path is not None
        assert simulate_y_target_path is not None

        x_source = pd.read_csv(simulate_x_source_path).to_numpy()
        y_source = pd.read_csv(simulate_y_source_path).to_numpy()
        x_target = pd.read_csv(simulate_x_target_path).to_numpy()
        y_target = pd.read_csv(simulate_y_target_path).to_numpy()

        result["source"] = ("source", x_source, y_source, x_source)
        result["target"] = ("target", x_target, y_target, x_target)

        return result

    def is_eligible(self, task: str) -> bool:
        return task == SIMULATION
