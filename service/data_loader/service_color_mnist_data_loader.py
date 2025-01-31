from service.data_loader.service_abstract_data_loader import AbstractDataLoaderService

from utils.constants import *

from torchvision import datasets
from typing import Dict, Tuple
import numpy as np


COLOR_FLIP_PROBS = [0.3, 0.5, 1.0]


class ColorMnistDataLoaderService(AbstractDataLoaderService):

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

        # Simulate covariate shift
        total_clients = len(COLOR_FLIP_PROBS)
        B = len(train_y) // total_clients
        for i, prob in enumerate(COLOR_FLIP_PROBS):
            x = train_x[i * B: i * B + B]
            y = train_y[i * B: i * B + B]
            client_x, client_y = self._make_environment(x, y, prob,
                                                        need_flatten)
            hospital_id = str(i)
            result[hospital_id] = (hospital_id, client_x, client_y)

        target_x, target_y = self._make_environment(test_x, test_y, 0,
                                                    need_flatten)
        result["target"] = ("target", target_x, target_y)

        return result

    def is_eligible(self, task: str) -> bool:
        return task == COLOR_MNIST

    """ Private methods """

    def _make_environment(self, images: np.ndarray,
                          labels: np.ndarray,
                          color_flip_prob: float,
                          need_flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        def bernoulli(p, size):
            return np.random.rand(size) < p

        def xor(a, b):
            # Assumes both inputs are either 0 or 1
            return np.absolute(a - b)

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]

        # Assign a binary label: larger than 5, label = 1
        labels = (labels >= 5).astype(float)

        # Assign a color based on the label;
        # Flip the color with probability p
        # e.g. Larger than 5, color = 1 with prob: 1 - p, color = 0 with prob: p (flip the color)
        colors = xor(labels, bernoulli(color_flip_prob, len(labels)))

        # Apply the color to the image by zeroing out the other color channel
        # N * channel * width * height
        images = np.stack([images, images], axis=1)

        # Larger than 5, channel: 1 - color = 0 set to 0, zero out red channel -> green digit
        N = len(images)
        zero_out_channel = (1 - colors).astype(int)

        # N * channel * width * height
        images[np.arange(N), zero_out_channel, :, :] *= 0

        if need_flatten:
            # N * (channel * width * height)
            images = images.reshape(images.shape[0], -1)

        return images / 255., labels[:, None]
