from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService

from utils.constants import *

from torchvision import datasets
from typing import Dict, Tuple
import numpy as np

from torchvision import transforms

ROTATE_ANGLES = ["0.0,60.0", "60.0,120.0", "120.0,180.0"]


class BinarizedMnistDataLoaderService(AbstractDataLoaderService):

    def __init__(self) -> None:
        # self._train_mnist = datasets.MNIST('~/datasets/mnist',
        #                                    train=True,
        #                                    download=True)
        # self._test_mnist = datasets.MNIST('~/datasets/mnist',
        #                                   train=False,
        #                                   download=True)
        pass

    """ Public methods """

    def load_data(self, **kwargs) -> Dict[str, Tuple[str, np.ndarray, np.ndarray]]:

        result: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = dict()

        need_flatten = kwargs.get("need_flatten")
        if need_flatten is None:
            need_flatten = True

        # Load data
        train_x = self._train_mnist.data.cpu().detach().numpy()
        train_y = self._train_mnist.targets.cpu().detach().numpy()
        test_x = self._test_mnist.data.cpu().detach().numpy()
        test_y = self._test_mnist.targets.cpu().detach().numpy()

        # Binarized
        # train_x[train_x <= 251] = 0.0
        # test_x[test_x <= 251] = 0.0
        # train_x[train_x > 251] = 1.0
        # test_x[test_x > 251] = 1.0

        train_x[train_x > 0.0] = 1.0
        test_x[test_x > 0.0] = 1.0

        # Simulate covariate shift
        total_clients = len(ROTATE_ANGLES)
        B = len(train_y) // total_clients
        for i, angle in enumerate(ROTATE_ANGLES):
            x = train_x[i * B: i * B + B]
            y = train_y[i * B: i * B + B]

            from_angle, to_angle = float(angle.split(',')[0]), float(angle.split(',')[1])
            client_x, client_y = self._make_environment(x, y,
                                                        from_angle, to_angle,
                                                        need_flatten)
            hospital_id = str(i)
            result[hospital_id] = (hospital_id, client_x, client_y)

        target_x, target_y = self._make_environment(test_x, test_y, 0, 0,
                                                    need_flatten)
        result["target"] = ("target", target_x, target_y)

        return result

    def is_eligible(self, task: str) -> bool:
        return task == BINARIZED_MNIST

    """ Private methods """

    def _make_environment(self, images: np.ndarray,
                          labels: np.ndarray,
                          from_angle: float,
                          to_angle: float,
                          need_flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        # Assign a binary label: larger than 5, label = 1
        labels = (labels >= 5).astype(float)

        rotation = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomRotation(
                                           degrees=(from_angle, to_angle)),
                                       transforms.ToTensor()])

        updated_images = np.zeros(
            (len(images), images.shape[1], images.shape[2]))
        for i in range(len(images)):
            updated_images[i] = rotation(images[i])

        if need_flatten:
            # N * (channel * width * height)
            updated_images = updated_images.reshape(
                updated_images.shape[0], -1)

        return updated_images, labels[:, None]
