from service.classification.service_abstract_classifier import AbstractClassifierService
from service.classification.service_avg_classifier import FedWeightAvgClassifierService
from service.classification.service_sgd_classifier import FedWeightSgdClassifierService


class ClassifierServiceFactory:

    """ Initialize """

    def __init__(self) -> None:
        self._classifiers = []
        children = AbstractClassifierService.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._classifiers.append(child())

    """ Public methods """

    def get_classifier(self, type: str) -> AbstractClassifierService:

        if len(self._classifiers) == 0:
            return None

        for classifier in self._classifiers:
            if classifier.is_eligible(type):
                return classifier

        return None
