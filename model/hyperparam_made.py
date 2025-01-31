class MadeHyperParam:

    """ Initialize """

    def __init__(self, made_epochs: int,
                 made_hiddens: str,
                 num_masks: int,
                 samples: int,
                 resample_every: int,
                 natural_ordering: bool,
                 made_learning_rate: float,
                 made_weight_decay: float,
                 batch_size: int) -> None:
        self._made_epochs = made_epochs
        self._made_hiddens = made_hiddens
        self._num_masks = num_masks
        self._samples = samples
        self._resample_every = resample_every
        self._natural_ordering = natural_ordering
        self._made_learning_rate = made_learning_rate
        self._made_weight_decay = made_weight_decay
        self._batch_size = batch_size

    """ Getters """

    @property
    def made_epochs(self) -> int:
        return self._made_epochs

    @property
    def made_hiddens(self) -> str:
        return self._made_hiddens

    @property
    def num_masks(self) -> int:
        return self._num_masks

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def resample_every(self) -> int:
        return self._resample_every

    @property
    def natural_ordering(self) -> bool:
        return self._natural_ordering

    @property
    def made_learning_rate(self) -> float:
        return self._made_learning_rate

    @property
    def made_weight_decay(self) -> float:
        return self._made_weight_decay

    @property
    def batch_size(self) -> int:
        return self._batch_size
