class VaeHyperParam:

    """ Initialize """

    def __init__(self, vae_epochs: int,
                 vae_latent_dim: int,
                 vae_hiddens: str,
                 vae_learning_rate: float,
                 vae_weight_decay: float,
                 batch_size: int) -> None:
        self._vae_epochs = vae_epochs
        self._vae_latent_dim = vae_latent_dim
        self._vae_hiddens = vae_hiddens
        self._vae_learning_rate = vae_learning_rate
        self._vae_weight_decay = vae_weight_decay
        self._batch_size = batch_size

    """ Getters """

    @property
    def vae_epochs(self) -> int:
        return self._vae_epochs

    @property
    def vae_latent_dim(self) -> int:
        return self._vae_latent_dim

    @property
    def vae_hiddens(self) -> str:
        return self._vae_hiddens

    @property
    def vae_learning_rate(self) -> float:
        return self._vae_learning_rate

    @property
    def vae_weight_decay(self) -> float:
        return self._vae_weight_decay

    @property
    def batch_size(self) -> int:
        return self._batch_size
