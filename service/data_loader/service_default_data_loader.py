from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService

from utils.utils_log import LogUtils
from utils.constants import *

from typing import Dict, Tuple
import pandas as pd
import numpy as np


class DefaultDataLoaderService(AbstractDataLoaderService):

    def load_data(self, **kwargs) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:

        result = dict()

        dataset_path = kwargs.get("dataset_path")
        time_dataset_path = kwargs.get("time_dataset_path")
        min_death_count = kwargs.get("min_death_count")
        hospital_id_col = kwargs.get("hospital_id_col")
        dedicated_hospital_ids = kwargs.get("dedicated_hospital_ids")
        task = kwargs.get("task")
        run_with_fl = kwargs.get("run_with_fl")

        assert dataset_path is not None
        assert dedicated_hospital_ids is not None
        assert time_dataset_path is not None
        assert min_death_count is not None
        assert hospital_id_col is not None
        assert task is not None

        icu_data = pd.read_csv(dataset_path)
        time_icu_data = pd.read_csv(time_dataset_path)
        LogUtils.instance().log_info(
            "Dataset shape: {}, time dataset shape: {}".format(icu_data.shape, time_icu_data.shape))

        # Select dedicated hospitals
        for idx, hospital_id in enumerate(dedicated_hospital_ids):
            hospital_data = icu_data[icu_data["hospitalid"] == hospital_id]
            time_hospital_data = time_icu_data[time_icu_data["hospitalid"] == hospital_id]

            time_x, time_y, x, y, los = self._process_data(hospital_data,
                                                           time_hospital_data,
                                                           task)
            if task == DEATH:
                if run_with_fl:
                    cls_x, cls_y = x, y
                else:
                    one_hot_hospital_ids = np.zeros((len(y), len(dedicated_hospital_ids)))
                    one_hot_hospital_ids[:, idx] = 1.0
                    hospital_x = np.concatenate((one_hot_hospital_ids, x), axis=1)
                    cls_x, cls_y = hospital_x, y

            elif task == LENGTH:
                if run_with_fl:
                    cls_x, cls_y = x, los
                else:
                    one_hot_hospital_ids = np.zeros((len(los), len(dedicated_hospital_ids)))
                    one_hot_hospital_ids[:, idx] = 1.0
                    hospital_x = np.concatenate((one_hot_hospital_ids, x), axis=1)
                    cls_x, cls_y = hospital_x, los

            else:
                if run_with_fl:
                    cls_x, cls_y = time_x, time_y
                else:
                    max_time_windows = int(time_hospital_data['time_window'].max()) + 1  # 6
                    one_hot_hospital_ids = np.zeros((len(y), max_time_windows, len(dedicated_hospital_ids)))
                    one_hot_hospital_ids[:, :, idx] = 1.0
                    hospital_time_x = np.concatenate((one_hot_hospital_ids, time_x), axis=2)
                    cls_x, cls_y = hospital_time_x, time_y

            density_x = x

            result[hospital_id] = (hospital_id, cls_x, cls_y, density_x)
            LogUtils.instance().log_info('Hospital: {}, Total patient count: {}, death count: {}'.format(
                int(hospital_id), len(hospital_data), np.sum(y)))

        return result

    def is_eligible(self, task: str) -> bool:
        return task != SIMULATION and task != UNSUPERVISED

    def _process_data(self, df, time_df, task):

        drug_time_data = []
        time_labels = []

        drug_original_data = []
        original_labels = []
        length_of_stays = []

        # Get unique patients and the maximum number of time windows
        original_patients = df['patientunitstayid'].unique()
        patients = time_df['patientunitstayid'].unique()
        assert set(original_patients) == set(patients)
        max_time_windows = int(time_df['time_window'].max()) + 1  # 6

        for patient in patients:

            if task != DEATH and task != LENGTH:
                # Time data
                patient_time_data = time_df[time_df['patientunitstayid'] == patient].sort_values('time_window')
                patient_drug_time_data = []
                patient_time_labels = []

                for time_window in range(max_time_windows):  # 0, 1, 2, 3, 4, 5
                    drug_time_window = time_window - 1  # -1, 0, 1, 2, 3, 4
                    current_data = patient_time_data[patient_time_data['time_window'] == drug_time_window]

                    if not current_data.empty:
                        time_x = current_data.iloc[0, 7:].to_list()
                    else:
                        # Pad with zeros if no data for the time window
                        time_x = [0.0] * 268
                    patient_drug_time_data.append(time_x)

                    # Labels are offset by 1 time window
                    next_data = patient_time_data[patient_time_data['time_window'] == time_window]
                    label = next_data.iloc[0][task] if not next_data.empty else 0
                    patient_time_labels.append(label)

                drug_time_data.append(patient_drug_time_data)
                time_labels.append(patient_time_labels)

            # Original data
            patient_original_data = df[df['patientunitstayid'] == patient]

            x = patient_original_data.iloc[:, 4:]
            y = patient_original_data.iloc[:, 2]
            los = patient_original_data.iloc[:, 3]

            drug_original_data.append(x)
            original_labels.append(y)
            length_of_stays.append(los)

        drug_time_data = np.array(drug_time_data, dtype=object)
        time_labels = np.array(time_labels, dtype=object)

        drug_original_data = np.array(drug_original_data, dtype=object)[:, 0, :]
        original_labels = np.array(original_labels, dtype=object)
        length_of_stays = np.array(length_of_stays, dtype=object)

        return drug_time_data, time_labels, drug_original_data, original_labels, length_of_stays
