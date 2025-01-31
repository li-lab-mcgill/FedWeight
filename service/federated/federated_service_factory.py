from service.federated.service_abstract_federated import AbstractFederatedService
from service.classification.service_abstract_classifier import AbstractClassifierService

from service.federated.service_avg_federated import FedWeightAvgService
from service.federated.service_sgd_federated import FedWeightSgdService


class FederatedServiceFactory:

    """ Initialize """

    def __init__(self, classifier_service: AbstractClassifierService) -> None:
        self._services = []
        children = AbstractFederatedService.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._services.append(child(classifier_service))

    """ Public methods """

    def get_service(self, type: str) -> AbstractFederatedService:

        if len(self._services) == 0:
            return None

        for service in self._services:
            if service.is_eligible(type):
                return service

        return None
