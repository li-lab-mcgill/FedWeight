class FederatedHyperParam:

    """ Initialize """

    def __init__(self, test_size: float,
                 val_size: float,
                 total_feature: int,
                 fl_hiddens: str,
                 learning_rate: float,
                 weight_decay: float,
                 total_round: int,
                 local_epochs: int,
                 batch_size: int) -> None:
        self._test_size = test_size
        self._val_size = val_size
        self._total_feature = total_feature
        self._fl_hiddens = fl_hiddens
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._total_round = total_round
        self._local_epochs = local_epochs
        self._batch_size = batch_size

    """ Getters """

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def val_size(self) -> float:
        return self._val_size

    @property
    def total_feature(self) -> int:
        return self._total_feature

    @property
    def fl_hiddens(self) -> str:
        return self._fl_hiddens

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def total_round(self) -> int:
        return self._total_round

    @property
    def local_epochs(self) -> int:
        return self._local_epochs

    @property
    def batch_size(self) -> int:
        return self._batch_size
