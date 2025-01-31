class FocalHyperParam:

    """ Initialize """

    def __init__(self, focal_alpha, focal_gamma):
        self._focal_alpha = focal_alpha
        self._focal_gamma = focal_gamma

    """ Getters """

    @property
    def focal_alpha(self):
        return self._focal_alpha

    @property
    def focal_gamma(self):
        return self._focal_gamma
