import numpy as np
from scipy.special import digamma, polygamma, gammaln
from copy import deepcopy


class LDA:

    def __init__(self, K: int = 3) -> None:
        """
        LDA with variational inference
        :param K: Topics
        :param N: Patients (documents)
        :param Mn: Drugs taken (word count) for the n-th patient (document)
        :param M: Array of drugs taken by the patients
        :param D: Unique drugs across all the patients
        :param tokens: Tokens
        :param gamma: variational parameter for document topic distribution theta: N x K
        :param lambda: variational parameter for word topic distribution beta: K x D
        :param phi: variational parameter for topic assignment z: N x M[n] x K
        :param alpha: dirichlet prior for document topic distribution theta: K x 1
        :param eta: dirichlet prior for word topic distribution beta: D x 1
        """
        self.K = K
        self.N = None
        self.D = None
        self.tokens = None
        self.M = None

        self.params = {
            'gamma': None,
            'lambda': None,
            'phi': None,
            'alpha': None,
            'eta': None
        }

    """ Public methods """

    def fit(self, x: np.ndarray, epochs: int = 10) -> None:

        self.N = x.shape[0]
        self.D = x.shape[1]
        self.M = np.sum(x, axis=1).astype(int)  # N x 1
        self.x = x

        # [1 1 0 1 0 1] -> [0 1 nan 2 nan 3]
        self.m_dict = np.empty((self.N, self.D))  # N x D
        self.m_dict[:] = np.nan
        for n in range(self.N):
            m = 0
            for d in range(self.D):
                if self.x[n][d] == 1:
                    self.m_dict[n][d] = m
                    m += 1

        self._init_params()

        for i in range(epochs):

            self._E_step()
            self._M_step()
            elbo = self._calculate_elbo()

            print('Epoch: {}, elbo: {}'.format(i, elbo))

    def predict(self, x: np.ndarray, epochs: int = 10):

        self.N = x.shape[0]
        self.D = x.shape[1]
        self.M = np.sum(x, axis=1).astype(int)  # N x 1
        self.x = x

        # [1 1 0 1 0 1] -> [0 1 nan 2 nan 3]
        self.m_dict = np.empty((self.N, self.D))  # N x D
        self.m_dict[:] = np.nan
        for n in range(self.N):
            m = 0
            for d in range(self.D):
                if self.x[n][d] == 1:
                    self.m_dict[n][d] = m
                    m += 1

        self._init_params(is_test=True)

        for i in range(epochs):

            self._E_step()
            elbo = self._calculate_elbo()

            print('Epoch: {}, elbo: {}'.format(i, elbo))

    """ Init """

    def _init_params(self, is_test=False) -> None:

        # Initialize variational parameters
        gamma = np.random.dirichlet(100 * np.ones(self.K), self.N)  # N x K
        lambda_ = np.random.dirichlet(100 * np.ones(self.D), self.K)  # K x D
        phi = np.array([np.random.dirichlet(100 * np.ones(self.K), Mn)
                       for Mn in self.M])  # N x M[n] x K

        # Initialize model parameters
        if not is_test:
            alpha = np.ones(self.K)  # K x 1
            eta = np.full(self.D, 0.0001)  # D x 1
        else:
            alpha = deepcopy(self.params['alpha'])  # K x 1
            eta = deepcopy(self.params['eta'])  # D x 1

        self.params = {
            'gamma': gamma,
            'lambda': lambda_,
            'phi': phi,
            'alpha': alpha,
            'eta': eta
        }

    """ EM """

    def _E_step(self) -> None:

        print("E Step")
        gamma = self._update_gamma()
        lambda_ = self._update_lambda()
        phi = self._update_phi()

        self.params['gamma'] = gamma
        self.params['lambda'] = lambda_
        self.params['phi'] = phi

    def _M_step(self) -> None:

        alpha = self._update_alpha()
        eta = self._update_eta()

        self.params['alpha'] = alpha
        self.params['eta'] = eta

    """ E Step """

    def _update_gamma(self) -> np.array:
        """
         $$\gamma_{nk} = \alpha_k + \sum_{m=1}^{M_n} \phi_{nmk}$$
        :return:
        """
        print("Update gamma")
        result = np.zeros((self.N, self.K))  # N x K

        alpha = deepcopy(self.params['alpha'])  # K x 1
        phi = deepcopy(self.params['phi'])  # N x M[n] x K

        for n in range(self.N):
            result[n] = alpha + np.sum(phi[n], axis=0)  # 1 x K

        # Normalization
        gamma_sum = np.sum(result, axis=1)[:, None]  # N x 1
        return result / gamma_sum

    def _update_lambda(self) -> np.array:
        """
        $$\lambda_{kd} = \eta_d + \sum_{n=1}^N \sum_{m=1}^{M_n}  \phi_{nmk} w_{nm}^d $$
        :return:
        """
        print("Update lambda")
        result = np.zeros((self.K, self.D))  # K x D

        eta = deepcopy(self.params['eta'])  # D x 1
        phi = deepcopy(self.params['phi'])  # N x M[n] x K

        for d in range(self.D):
            phi_sum = 0
            for n in range(self.N):
                if self.x[n][d] == 1:
                    m = int(self.m_dict[n][d])
                    assert not np.isnan(m)
                    phi_sum += phi[n][m]  # K x 1

            result[:, d] = eta[d] + phi_sum  # K x 1

        # Normalization
        lambda_sum = np.sum(result, axis=1)[:, None]  # K x 1
        return result / lambda_sum

    def _update_phi(self) -> np.array:
        """
        $$\phi_{nmk} \ltimes \exp(\Psi(\gamma_{nk}) - Psi(\sum_{j=1}^K \gamma_{nk})
        +\sum_{d=1}^D x_{nm}^d  \Psi(\lambda_{kd}) - \Psi(\sum_{j=1}^D \lambda_{kj}))$$
        :return:
        """
        print("Update phi")
        result = np.array([np.ones((n, self.K))
                          for n in self.M], dtype=object)  # N x M[n] x K

        gamma = deepcopy(self.params['gamma'])  # N x K
        lambda_ = deepcopy(self.params['lambda'])  # K x D

        for n in range(self.N):
            for m in range(self.M[n]):

                gamma_n = gamma[n]  # K x 1
                gamma_sum = np.sum(gamma_n)  # 1 x 1
                gamma_diff = digamma(gamma_n) - digamma(gamma_sum)  # K x 1

                lambda_sum = np.sum(lambda_, axis=1)[:, None]  # K x 1
                lambda_diff = digamma(lambda_) - digamma(lambda_sum)  # K x D

                x_nm = np.zeros(self.D)
                drug_ids = np.where(self.x[n] == 1)[0]  # All drugs taken
                drug_id = drug_ids[m]  # Drug id taken
                x_nm[drug_id] = 1

                term_2 = np.sum(x_nm * lambda_diff, axis=1)  # K x 1

                a = gamma_diff + term_2  # K x 1
                a = np.where(a > 20, 20, a)
                phi = np.exp(a)
                phi_sum = np.sum(phi)
                result[n][m] = phi / phi_sum

        return result


    """ M Step """

    def _update_alpha(self, max_iter: int = 1000, tol: float = 0.1) -> np.array:
        """
        $$\alpha_{new} = \alpha_{old} - H(\alpha_{old})^{-1} g(\alpha_{old})$$
        """
        print("Update alpha")
        alpha = deepcopy(self.params['alpha'])
        gamma = deepcopy(self.params['gamma'])

        for _ in range(max_iter):

            alpha_old = alpha

            # First derivative
            # digamma(np.sum(gamma, axis=1))[:, None] # N x 1
            g = self.N * (digamma(np.sum(alpha)) - digamma(alpha)) + \
                np.sum(digamma(gamma) - digamma(np.sum(gamma, axis=1))
                       [:, None], axis=0)  # 1 x K

            # Second derivative
            h = -1 * self.N * polygamma(1, alpha)
            z = self.N * polygamma(1, np.sum(alpha))
            c = np.sum(g / h) / (z ** (-1.0) + np.sum(h ** (-1.0)))

            # Update alpha
            alpha = alpha - (g - c) / h

            # Stopping criteria
            if np.sqrt(np.mean(np.square(alpha - alpha_old))) < tol:
                break

        return alpha

    def _update_eta(self, max_iter: int = 100, tol: int = 0.1) -> np.array:
        """
        $$\eta_{new} = \eta_{old} - H(\eta_{old})^{-1}g(\eta_{old})$$
        """
        print("Update eta")
        eta = deepcopy(self.params['eta'])  # D x 1
        lambda_ = deepcopy(self.params['lambda'])  # K x D

        for i in range(max_iter):

            if i % 5 == 0:
                print("Iteration: {}, eta: {}".format(i, eta))

            eta_old = eta

            # First derivative
            # digamma(np.sum(lambda_, axis=1)) # K x 1
            # digamma(np.sum(lambda_, axis=1))[:, None] # K x 1
            # g = self.K * (digamma(np.sum(eta)) - digamma(eta)) + \
            #     np.sum(digamma(lambda_) - digamma(np.sum(lambda_, axis=1))
            #            [:, None], axis=0)  # 1 x D

            g = self.K * (digamma(np.sum(eta)) - digamma(eta)) + \
                np.sum(
                    digamma(lambda_) - np.tile(digamma(np.sum(lambda_, axis=1)), (self.D, 1)).T,
                    axis=0
                )

            # Second derivative
            h = -1 * self.K * polygamma(1, eta)
            z = self.K * polygamma(1, np.sum(eta))
            c = np.sum(g / h) / (z ** (-1.0) + np.sum(h ** (-1.0)))

            # Update eta
            # print("g {} c {} h {} (g - c) / h {}".format(g, c, h, ((g - c) / h)))
            eta = eta - (g - c) / h

            # Stopping criteria
            if np.sqrt(np.mean(np.square(eta - eta_old))) < tol:
                break

        return eta

    """ ELBO """

    def _calculate_elbo(self) -> float:

        print("Evaluate ELBO")

        # Expectation for p
        p_z_given_theta = self._p_z_given_theta()
        p_theta_given_alpha = self._p_theta_given_alpha()
        p_beta_given_eta = self._p_beta_given_eta()
        p_x_given_z_and_beta = self._p_x_given_z_and_beta()

        # Expectation for q
        q_theta_given_gamma = self._q_theta_given_gamma()
        q_z_given_phi = self._q_z_given_phi()
        q_beta_given_lambda = self._q_beta_given_lambda()

        # ELBO
        return p_z_given_theta + p_theta_given_alpha + p_beta_given_eta + p_x_given_z_and_beta - q_theta_given_gamma - q_z_given_phi - q_beta_given_lambda

    def _p_z_given_theta(self) -> float:

        result = 0

        phi = deepcopy(self.params['phi'])  # N x M[n] x K
        gamma = deepcopy(self.params['gamma'])  # N x K

        for n in range(self.N):

            phi_n = phi[n]  # M[n] x K
            gamma_n = gamma[n]  # 1 x K

            for m in range(self.M[n]):

                gamma_sum = np.sum(gamma_n)
                gamma_diff = digamma(gamma_n) - digamma(gamma_sum)  # 1 x K

                result += np.sum(phi_n[m] * gamma_diff)

        return result

    def _p_theta_given_alpha(self) -> float:

        result = 0

        alpha = deepcopy(self.params['alpha'])  # K
        gamma = deepcopy(self.params['gamma'])  # N x K

        gamma_sum = np.sum(gamma, axis=1)[:, None]  # N x 1
        gamma_diff = digamma(gamma) - digamma(gamma_sum)  # N x K
        term_1 = np.sum((alpha[None, :] - 1) * gamma_diff)  # 1 x 1

        result = term_1 + gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
        return result

    def _p_beta_given_eta(self) -> float:

        result = 0

        eta = deepcopy(self.params['eta'])  # D
        lambda_ = deepcopy(self.params['lambda'])  # K x D

        lambda_sum = np.sum(lambda_, axis=1)[:, None]  # K x 1
        lambda_diff = digamma(lambda_) - digamma(lambda_sum)  # K x D
        term_1 = np.sum((eta[None, :] - 1) * lambda_diff)

        result = term_1 + self.K * \
            gammaln(np.sum(eta)) - self.K * np.sum(gammaln(eta))

        return result

    def _p_x_given_z_and_beta(self) -> float:

        result = 0

        phi = deepcopy(self.params['phi'])  # N x M[n] x K
        lambda_ = deepcopy(self.params['lambda'])  # K x D

        for n in range(self.N):

            x_n = self.x[n][None, :]  # 1 x D

            for m in range(self.M[n]):

                phi_nm = phi[n][m][:, None]  # K x 1

                lambda_sum = np.sum(lambda_, axis=1)[:, None]  # K x 1
                lambda_diff = digamma(lambda_) - digamma(lambda_sum)  # K x D

                result += np.sum(phi_nm * x_n * lambda_diff)

        return result

    def _q_theta_given_gamma(self) -> float:

        result = 0

        gamma = deepcopy(self.params['gamma'])  # N x K

        gamma_sum = np.sum(gamma, axis=1)[:, None]  # N x 1
        gamma_diff = digamma(gamma) - digamma(gamma_sum)  # N x K

        term_1 = (gamma - 1) * gamma_diff  # N x K
        term_2 = gammaln(np.sum(gamma, axis=1))  # N x 1
        term_3 = np.sum(gammaln(gamma), axis=1)  # N x 1

        result = np.sum(term_1 + term_2 - term_3)

        return result

    def _q_z_given_phi(self) -> float:

        result = 0

        phi = deepcopy(self.params['phi'])  # N x M[n] x K

        for n in range(self.N):
            for m in range(self.M[n]):

                phi_mn = phi[n][m]  # K x 1
                result += np.sum(phi_mn * np.log(phi_mn))

        return result

    def _q_beta_given_lambda(self) -> float:

        result = 0

        lambda_ = deepcopy(self.params['lambda'])  # K x D

        lambda_sum = np.sum(lambda_, axis=1)  # K x 1
        lambda_diff = digamma(lambda_) - digamma(lambda_sum)  # K x D

        term_1 = (lambda_ - 1) * lambda_diff  # K x D
        term_2 = gammaln(np.sum(lambda_, axis=1))  # K x 1
        term_3 = np.sum(gammaln(lambda_), axis=1)  # K x 1

        result = term_1 + term_2 - term_3

        return result
