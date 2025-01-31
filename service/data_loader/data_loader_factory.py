from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService
from service.data_loader.service_default_data_loader import DefaultDataLoaderService
from service.data_loader.service_unsupervised_data_loader import UnsupervisedDataLoaderService
from service.data_loader.service_simulate_data_loader import SimulateDataLoaderService


class DataLoaderFactory:
    """ Initialize """

    def __init__(self) -> None:
        self._data_loaders = []
        children = AbstractDataLoaderService.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._data_loaders.append(child())

    """ Public methods """

    def get_data_loader(self, task: str) -> AbstractDataLoaderService:

        if len(self._data_loaders) == 0:
            return None

        for data_loader in self._data_loaders:
            if data_loader.is_eligible(task):
                return data_loader

        return None
