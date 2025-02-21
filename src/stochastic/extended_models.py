"""
sde_models_extended.py

Extends the 12 SDE models from 'sde_models.py' to support:
  1) sample_paths_and_derivative(...) -> (S, dSdS0)
     so we can do pathwise Delta for a European call.
  2) pathwise_delta_eurocall(...) -> uses the above derivative
  3) implied_vol_mc(...) -> bisection on MC price to find IV.

Each class shows how to compute partial(S_{t+1}, S_t) by chain rule
and thus partial(S_{t+1}, S0).
"""

from typing import Optional, Tuple, Callable
from src.stochastic.sde.base_sde_extended import BaseModelExtended
import numpy as np
import math


###############################################################################
# 1) Black–Scholes–Merton
###############################################################################
class BlackScholesMerton(BaseModelExtended):
    """
    dS = S*(r - q)*dt + S*sigma dW.
    We'll do a log-Euler approach for sample_paths.

    Derivative logic:
      S_{k+1} = S_k * exp( (r-q-0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
      => partial(S_{k+1}, S_k) = S_{k+1}/S_k
      => partial(S_{k+1}, S0) = partial(S_{k+1}, S_k)*partial(S_k, S0).
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        S0: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step])
            log_s_next = log_s_k + drift + vol * Z
            S[:, step + 1] = np.exp(log_s_next)
        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0

        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            ratio = S_next / (S[:, step] + 1e-16)

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dSdS0[:, step] * ratio
        return S, dSdS0

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}"


###############################################################################
# 2) MertonJump
###############################################################################
class MertonJump(BaseModelExtended):
    """
    Merton jump: S_{k+1} = S_k * exp(...) * (product of jump factors if Poisson>0).

    If jumps are independent of S0, partial(S_{k+1}, S_k) = S_{k+1}/S_k.

    This is the same ratio approach as BSM, ignoring jump dependence on S0.
    """

    def __init__(self, r, q, sigma, S0, jump_intensity, muJ, sigmaJ, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma
        self.lambda_ = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        EJ = math.exp(self.muJ + 0.5 * self.sigmaJ**2)
        drift_corr = (self.r - self.q) - self.lambda_ * (EJ - 1)
        drift = (drift_corr - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)

            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_j[i])
                    S_next[i] *= math.exp(Y.sum())
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0

        EJ = math.exp(self.muJ + 0.5 * self.sigmaJ**2)
        drift_corr = (self.r - self.q) - self.lambda_ * (EJ - 1)
        drift = (drift_corr - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            ratio = S_next / (S[:, step] + 1e-16)

            # apply ratio to derivative first
            dS_temp = dSdS0[:, step] * ratio

            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_j[i])
                    jump_factor = math.exp(Y.sum())
                    S_next[i] *= jump_factor
                    dS_temp[i] *= jump_factor

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp
        return S, dSdS0

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}, MertonJump(sigma={self.sigma}, lambda={self.lambda_}, "
            f"muJ={self.muJ}, sigmaJ={self.sigmaJ})"
        )


###############################################################################
# 3) DupireLocalVol
###############################################################################
class DupireLocalVol(BaseModelExtended):
    """
    S_{k+1} = S_k + (r-q)*S_k dt + sigma_fn(S_k,t_k)*S_k sqrt(dt)*Z  (or a log form).
    We'll do a simpler Euler version:
       S_{k+1} = S_k + mu(S_k,t_k)*dt + sigma(S_k,t_k)* sqrt(dt)*Z.

    partial(S_{k+1}, S_k) = 1 + partial(mu(S_k,t_k),S_k)*dt + partial(sigma(S_k,t_k),S_k)* sqrt(dt)*Z.
    Then chain with partial(S_k, S0).
    """

    def __init__(
        self, r, q, S0, sigma_fn: Callable[[float, float], float], random_state=None
    ):
        super().__init__(r, q, S0, random_state)
        self.sigma_fn = sigma_fn

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0

        for step in range(n_steps):
            t_cur = step * dt
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            mu = (self.r - self.q) * S_k  # basic drift
            sig = np.array([self.sigma_fn(sk, t_cur) * sk for sk in S_k])
            S_next = S_k + mu * dt + sig * np.sqrt(dt) * Z
            S_next = np.clip(S_next, 1e-16, None)
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0

        for step in range(n_steps):
            t_cur = step * dt
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            dS_k = dSdS0[:, step]

            mu = (self.r - self.q) * S_k
            sig = np.array([self.sigma_fn(sk, t_cur) * sk for sk in S_k])

            # partial of [S_k + mu dt + sig sqrt(dt)*Z] wrt S_k
            # = 1 + partial(mu, sk)*dt + partial(sig, sk)* sqrt(dt)*Z
            # partial(mu, sk) = (r-q)
            # partial(sig, sk) = partial( sigma_fn(sk,t_cur)*sk , sk)
            # = sigma_fn'(sk) * sk + sigma_fn(sk)
            # We'll approximate sigma_fn'(sk) ~ 0 in example or do a small FD if needed.
            # for demonstration, let's skip derivative of sigma_fn wrt s => assume only linear in s for clarity
            # or set derivative_of_sigma_fn=0
            derivative_of_sigma_fn = 0.0
            # so partial(sig, sk) = sigma_fn(sk,t_cur)

            partial_mu = self.r - self.q
            partial_sig = np.array(
                [derivative_of_sigma_fn * sk + self.sigma_fn(sk, t_cur) for sk in S_k]
            )

            ratio = 1.0 + partial_mu * dt + partial_sig * np.sqrt(dt) * Z

            S_next = S_k + mu * dt + sig * np.sqrt(dt) * Z
            S_next = np.clip(S_next, 1e-16, None)

            dS_temp = dS_k * ratio

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp

        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, DupireLocalVol(sigma_fn=...)"


###############################################################################
# 4) CEV
###############################################################################
class CEV(BaseModelExtended):
    """
    S_{k+1} = S_k + (r-q)S_k dt + c*(S_k^gamma)* sqrt(dt)*Z.

    partial(S_{k+1}, S_k) = 1 + (r-q) dt + c*g* S_k^(gamma-1)* sqrt(dt)*Z.
    Then chain it.
    """

    def __init__(self, r, q, S0, c, gamma, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.c = c
        self.gamma = gamma

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        drift = self.r - self.q

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            S_next = (
                S_k + drift * S_k * dt + self.c * (S_k**self.gamma) * math.sqrt(dt) * Z
            )
            S_next = np.clip(S_next, 1e-16, None)
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        drift = self.r - self.q

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            dS_k = dSdS0[:, step]

            increment = drift * S_k * dt
            stoch_part = self.c * (S_k**self.gamma) * math.sqrt(dt) * Z
            S_next = S_k + increment + stoch_part
            S_next = np.clip(S_next, 1e-16, None)

            # partial(increment, S_k)= drift*dt
            # partial(stoch_part, S_k)= c*( gamma*S_k^(gamma-1) )* sqrt(dt)*Z
            partial_inc = drift * dt
            partial_sto = (
                self.c * (self.gamma * (S_k ** (self.gamma - 1))) * math.sqrt(dt)
            )
            ratio = 1.0 + partial_inc + partial_sto * Z
            dS_temp = dS_k * ratio

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp

        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, CEV(c={self.c}, gamma={self.gamma})"


###############################################################################
# 5) Heston
###############################################################################
class Heston(BaseModelExtended):
    """
    2D: dv = kappa(theta - v)dt + sigma_v sqrt(v) dWv
         dS = S*(r-q)dt + S* sqrt(v) dWs
    Corr(dWv, dWs)=rho.

    We'll store only S in the result. For partial derivative, ignoring v0's partial wrt S0,
    we assume S_t depends on S0 but not v0. Actually, a chain rule includes partial(v, S0)=0
    in standard approach if v is independent. We do a log-Euler approach for S ignoring partial wrt v0.

    partial(S_{k+1}, S_k) ~ e^(...)? We do the discrete step:
      S_{k+1} = S_k * exp( (r-q - 0.5*v_k)*dt + sqrt(v_k)* sqrt(dt)* Z_s )

    => partial(S_{k+1}, S_k)= S_{k+1}/S_k if v_k is not a function of S_k.
    However, in a strict sense, v might not be impacted by S0. We'll do the ratio approach again.
    """

    def __init__(self, r, q, S0, v0, kappa, theta, sigma_v, rho, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        v_arr = np.full(n_sims, self.v0, dtype=float)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, 0, None)

            s_old = S[:, step]
            drift = (self.r - self.q) * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s
            s_new = s_old * np.exp(drift + diff)

            S[:, step + 1] = s_new
            v_arr = v_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        v_arr = np.full(n_sims, self.v0, dtype=float)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # update v
            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, 0, None)

            s_old = S[:, step]
            ds_old = dSdS0[:, step]

            drift = (self.r - self.q) * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s

            s_new = s_old * np.exp(drift + diff)
            ratio = s_new / (s_old + 1e-16)

            S[:, step + 1] = s_new
            dSdS0[:, step + 1] = ds_old * ratio

            v_arr = v_next
        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return (
            f"{base}, Heston(v0={self.v0}, kappa={self.kappa}, "
            f"theta={self.theta}, sigma_v={self.sigma_v}, rho={self.rho})"
        )


###############################################################################
# 6) KouJump
###############################################################################
class KouJump(BaseModelExtended):
    """
    dS= S( (r-q)- lambda*(EJ-1) -0.5*sigma^2 )dt + ... + Poisson jumps w/ double-exponential.
    partial(S_{k+1}, S_k)= S_{k+1}/S_k if jumps are independent of S0.

    Similar to Merton, but jump distribution differs. ratio approach again.
    """

    def __init__(self, r, q, S0, sigma, lambda_, p_up, eta1, eta2, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma
        self.lambda_ = lambda_
        self.p_up = p_up
        self.eta1 = eta1
        self.eta2 = eta2
        # E[J] = p_up*(eta1/(eta1-1)) + (1-p_up)*(eta2/(eta2+1))
        # assume eta1>1, eta2>0
        EJ_up = self.eta1 / (self.eta1 - 1.0) if self.eta1 > 1 else math.inf
        EJ_dn = self.eta2 / (self.eta2 + 1.0) if self.eta2 > 0 else math.inf
        self.ej = self.p_up * EJ_up + (1 - self.p_up) * EJ_dn

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        drift_corr = (self.r - self.q) - self.lambda_ * (self.ej - 1)
        drift = (drift_corr - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s = np.log(S[:, step] + 1e-16)
            log_s_next = log_s + drift + vol * Z
            S_next = np.exp(log_s_next)
            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    for _ in range(N_j[i]):
                        u = self._rng.random()
                        if u < self.p_up:
                            x = self._rng.exponential(1.0 / self.eta1)
                            S_next[i] *= math.exp(x)
                        else:
                            x = self._rng.exponential(1.0 / self.eta2)
                            S_next[i] *= math.exp(-x)
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        drift_corr = (self.r - self.q) - self.lambda_ * (self.ej - 1)
        drift = (drift_corr - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s = np.log(S[:, step] + 1e-16)
            log_s_next = log_s + drift + vol * Z
            S_next = np.exp(log_s_next)
            ratio = S_next / (S[:, step] + 1e-16)

            dS_temp = dSdS0[:, step] * ratio

            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    for _ in range(N_j[i]):
                        u = self._rng.random()
                        if u < self.p_up:
                            x = self._rng.exponential(1.0 / self.eta1)
                            jump_factor = math.exp(x)
                        else:
                            x = self._rng.exponential(1.0 / self.eta2)
                            jump_factor = math.exp(-x)
                        S_next[i] *= jump_factor
                        dS_temp[i] *= jump_factor

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp

        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return (
            f"{base}, KouJump(sigma={self.sigma}, lambda={self.lambda_}, "
            f"p_up={self.p_up}, eta1={self.eta1}, eta2={self.eta2})"
        )


###############################################################################
# 7) Bates
###############################################################################
class Bates(BaseModelExtended):
    """
    Heston + lognormal jumps. partial(S_{k+1}, S_k)= S_{k+1}/S_k ignoring v->S0 again.
    """

    def __init__(
        self,
        r,
        q,
        S0,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        jump_intensity,
        muJ,
        sigmaJ,
        random_state=None,
    ):
        super().__init__(r, q, S0, random_state)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.lambda_ = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.ej = math.exp(muJ + 0.5 * sigmaJ**2)

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        v_arr = np.full(n_sims, self.v0)
        sqrt_dt = math.sqrt(dt)
        drift_corr = (self.r - self.q) - self.lambda_ * (self.ej - 1)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, 0, None)

            s_old = S[:, step]
            drift = drift_corr * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s
            s_temp = s_old * np.exp(drift + diff)

            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_j[i])
                    s_temp[i] *= math.exp(Y.sum())

            S[:, step + 1] = s_temp
            v_arr = v_next
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        v_arr = np.full(n_sims, self.v0)
        sqrt_dt = math.sqrt(dt)
        drift_corr = (self.r - self.q) - self.lambda_ * (self.ej - 1)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, 0, None)

            s_old = S[:, step]
            ds_old = dSdS0[:, step]

            drift = drift_corr * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s
            s_temp = s_old * np.exp(drift + diff)
            ratio = s_temp / (s_old + 1e-16)

            dS_temp = ds_old * ratio

            # jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_j[i])
                    jump_factor = math.exp(Y.sum())
                    s_temp[i] *= jump_factor
                    dS_temp[i] *= jump_factor

            S[:, step + 1] = s_temp
            dSdS0[:, step + 1] = dS_temp
            v_arr = v_next
        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, Bates(...)"


###############################################################################
# 8) SABR
###############################################################################
class SABR(BaseModelExtended):
    """
    Storing a minimal approach:
      F_{k+1}= ...
      alpha_{k+1}= ...

    partial(F_{k+1}, F_k) can be ~1 + ... ignoring complexities if small dt.
    We'll do a simplistic approach.
    """

    def __init__(self, r, q, S0, alpha0, beta, nu, rho, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.alpha0 = alpha0
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        F = np.zeros((n_sims, n_steps + 1))
        F[:, 0] = self.S0
        alpha_arr = np.full(n_sims, self.alpha0)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_s = Z1
            Z_a = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # alpha
            alpha_next = alpha_arr + self.nu * alpha_arr * sqrt_dt * Z_a
            alpha_next = np.clip(alpha_next, 1e-16, None)

            # F
            F_old = F[:, step]
            # simple euler => F_new= F_old+ (r-q)*F_old dt + alpha_arr F_old^beta sqrt(dt)* Z_s
            # ignoring 0.5 alpha^2 dt correction for brevity
            drift = (self.r - self.q) * F_old * dt
            stoch = alpha_arr * (F_old**self.beta) * sqrt_dt * Z_s
            F_new = F_old + drift + stoch
            F_new = np.clip(F_new, 1e-16, None)

            F[:, step + 1] = F_new
            alpha_arr = alpha_next

        return F

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        F = np.zeros((n_sims, n_steps + 1))
        dFdS0 = np.zeros((n_sims, n_steps + 1))
        F[:, 0] = self.S0
        dFdS0[:, 0] = 1.0
        alpha_arr = np.full(n_sims, self.alpha0)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_s = Z1
            Z_a = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # alpha
            alpha_next = alpha_arr + self.nu * alpha_arr * sqrt_dt * Z_a
            alpha_next = np.clip(alpha_next, 1e-16, None)

            F_old = F[:, step]
            dF_old = dFdS0[:, step]

            drift = (self.r - self.q) * F_old * dt
            # partial wrt F_old => (r-q)* dt + partial( alpha_arr F_old^beta, F_old ) * sqrt(dt)*Z_s
            # partial( F_old^beta, F_old)= beta F_old^(beta-1)
            # ignoring partial(alpha_arr, F_old) => alpha doesn't depend on F in our simplistic approach
            partial_stoch = (
                alpha_arr * self.beta * (F_old ** (self.beta - 1)) * sqrt_dt * Z_s
            )
            ratio = 1.0 + (self.r - self.q) * dt + partial_stoch

            F_new = F_old + drift + alpha_arr * (F_old**self.beta) * sqrt_dt * Z_s
            F_new = np.clip(F_new, 1e-16, None)
            dF_temp = dF_old * ratio

            F[:, step + 1] = F_new
            dFdS0[:, step + 1] = dF_temp
            alpha_arr = alpha_next

        return F, dFdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, SABR(alpha0={self.alpha0}, beta={self.beta}, nu={self.nu}, rho={self.rho})"


###############################################################################
# 9) VarianceGamma
###############################################################################
class VarianceGamma(BaseModelExtended):
    """
    S_t ~ S0 * exp( (r-q)*t + X_t ), X_t is a VG process with params (sigma, theta, nu).
    We'll do discrete steps in gamma subordinator approach.

    partial(S_{k+1}, S_k)= S_{k+1}/S_k
    => chain rule same ratio approach.
    """

    def __init__(self, r, q, S0, sigma, theta, nu, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma
        self.theta = theta
        self.nu = nu

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        mu = self.r - self.q

        for step in range(n_steps):
            # gamma increments G ~ Gamma(shape= dt/nu, scale= nu)
            G = self._rng.gamma(shape=dt / self.nu, scale=self.nu, size=n_sims)
            Z = self._rng.normal(size=n_sims)
            dX = self.theta * G + self.sigma * np.sqrt(G) * Z
            S_old = S[:, step]
            S_new = S_old * np.exp(mu * dt + dX)  # simplified
            S[:, step + 1] = S_new
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        mu = self.r - self.q

        for step in range(n_steps):
            G = self._rng.gamma(shape=dt / self.nu, scale=self.nu, size=n_sims)
            Z = self._rng.normal(size=n_sims)
            dX = self.theta * G + self.sigma * np.sqrt(G) * Z

            s_old = S[:, step]
            ds_old = dSdS0[:, step]
            s_new = s_old * np.exp(mu * dt + dX)
            ratio = s_new / (s_old + 1e-16)

            S[:, step + 1] = s_new
            dSdS0[:, step + 1] = ds_old * ratio
        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, VarianceGamma(sigma={self.sigma}, theta={self.theta}, nu={self.nu})"


###############################################################################
# 10) CGMY
###############################################################################
class CGMY(BaseModelExtended):
    """
    Placeholder for a heavy advanced model. partial(S_{k+1}, S_k)= S_{k+1}/S_k
    if the increment is add to the exponent. We do a naive approach.
    """

    def __init__(self, r, q, S0, C, G, M, Y, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

    def sample_paths(self, T, n_sims, n_steps):
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dt = T / n_steps
        for step in range(n_steps):
            # advanced approach. We'll do a dummy 0 increment for demonstration
            inc = np.zeros(n_sims)
            s_old = S[:, step]
            s_new = s_old * np.exp((self.r - self.q) * dt + inc)
            S[:, step + 1] = s_new
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        dt = T / n_steps
        for step in range(n_steps):
            inc = np.zeros(n_sims)
            s_old = S[:, step]
            ds_old = dSdS0[:, step]
            s_new = s_old * np.exp((self.r - self.q) * dt + inc)
            ratio = s_new / (s_old + 1e-16)

            S[:, step + 1] = s_new
            dSdS0[:, step + 1] = ds_old * ratio
        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, CGMY(C={self.C}, G={self.G}, M={self.M}, Y={self.Y})"


###############################################################################
# 11) NIG
###############################################################################
class NIG(BaseModelExtended):
    """
    Normal Inverse Gaussian. We'll do a stub approach. partial(S_{k+1}, S_k)= S_{k+1}/S_k if the increment is exponent additive.
    """

    def __init__(self, r, q, S0, alpha, beta, delta, random_state=None):
        super().__init__(r, q, S0, random_state)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def sample_paths(self, T, n_sims, n_steps):
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dt = T / n_steps
        for step in range(n_steps):
            # big simplification
            inc = np.zeros(n_sims)
            s_old = S[:, step]
            s_new = s_old * np.exp((self.r - self.q) * dt + inc)
            S[:, step + 1] = s_new
        return S

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        S = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0
        dt = T / n_steps
        for step in range(n_steps):
            inc = np.zeros(n_sims)
            s_old = S[:, step]
            ds_old = dSdS0[:, step]
            s_new = s_old * np.exp((self.r - self.q) * dt + inc)
            ratio = s_new / (s_old + 1e-16)

            S[:, step + 1] = s_new
            dSdS0[:, step + 1] = ds_old * ratio
        return S, dSdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, NIG(alpha={self.alpha}, beta={self.beta}, delta={self.delta})"


###############################################################################
# 12) OrnsteinUhlenbeck
###############################################################################
class OrnsteinUhlenbeck(BaseModelExtended):
    """
    If we treat S as X in OU.
    X_{k+1}= X_k + theta*(mu- X_k)*dt + sigma sqrt(dt)*Z
    partial(X_{k+1}, X_k)= 1 - theta dt

    Then partial(X_{k+1}, S0)= chain rule.
    But watch if you do a call payoff on X or transform X-> e^X, etc.
    We'll do direct X but that might not be a real 'price' if it can be negative.
    """

    def __init__(self, r, q, X0, theta, mu, sigma, random_state=None):
        super().__init__(r, q, X0, random_state)
        self.th = theta
        self.mu = mu
        self.sig = sigma

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        X = np.zeros((n_sims, n_steps + 1))
        X[:, 0] = self.S0
        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            X_k = X[:, step]
            X_next = X_k + self.th * (self.mu - X_k) * dt + self.sig * math.sqrt(dt) * Z
            X[:, step + 1] = X_next
        return X

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        X = np.zeros((n_sims, n_steps + 1))
        dXdS0 = np.zeros((n_sims, n_steps + 1))
        X[:, 0] = self.S0
        dXdS0[:, 0] = 1.0
        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            X_k = X[:, step]
            dX_k = dXdS0[:, step]
            partial_step = 1.0 - self.th * dt
            X_next = X_k + self.th * (self.mu - X_k) * dt + self.sig * math.sqrt(dt) * Z
            dX_temp = dX_k * partial_step
            X[:, step + 1] = X_next
            dXdS0[:, step + 1] = dX_temp
        return X, dXdS0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, OrnsteinUhlenbeck(theta={self.th}, mu={self.mu}, sigma={self.sig})"


###############################################################################
# Common Utility: pathwise Delta for a Euro call
###############################################################################
def pathwise_delta_eurocall(
    model: BaseModelExtended, strike: float, T: float, n_sims: int, n_steps: int
) -> float:
    """
    Delta = e^{-rT} * E[ 1_{S_T > strike} * partial(S_T, S0) ].

    For each model, sample_paths_and_derivative() must be implemented.

    Return scalar float Delta.
    """
    S, dSdS0 = model.sample_paths_and_derivative(T, n_sims, n_steps)
    ST = S[:, -1]
    dSTdS0 = dSdS0[:, -1]
    in_the_money = ST > strike
    payoff_deriv = in_the_money * dSTdS0
    disc = math.exp(-model.r * T)
    return disc * payoff_deriv.mean()


###############################################################################
# Common Utility: Monte Carlo price for Euro call
###############################################################################
def mc_price_eurocall(
    model: BaseModelExtended, strike: float, T: float, n_sims: int, n_steps: int
) -> float:
    """
    Price = e^{-rT} E[max(S_T- strike, 0)].
    """
    S = model.sample_paths(T, n_sims, n_steps)
    ST = S[:, -1]
    payoff = np.maximum(ST - strike, 0.0)
    return math.exp(-model.r * T) * payoff.mean()


###############################################################################
# Common Utility: implied_vol by bisection
###############################################################################
def implied_vol_mc(
    model_class: Callable[..., BaseModelExtended],
    r: float,
    q: float,
    S0: float,
    strike: float,
    T: float,
    market_price: float,
    n_sims: int = 50_000,
    n_steps: int = 50,
    seed: int = 123,
    vol_low: float = 1e-6,
    vol_high: float = 3.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Bisection to find sigma st. mc_price_eurocall(...) matches market_price.

    model_class must accept: (r, q, sigma, S0, random_state=seed)
    """
    for _ in range(max_iter):
        mid = 0.5 * (vol_low + vol_high)
        model = model_class(r=r, q=q, sigma=mid, S0=S0, random_state=seed)
        price_mid = mc_price_eurocall(model, strike, T, n_sims, n_steps)
        diff = price_mid - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            vol_high = mid
        else:
            vol_low = mid
    return 0.5 * (vol_low + vol_high)


###############################################################################
# Example main usage
###############################################################################
def main():
    # Quick demonstration: MertonJump, compute pathwise Delta and IV
    r = 0.05
    q = 0.02
    S0 = 100.0
    sigma = 0.20
    jump_intensity = 0.3
    muJ = 0.0
    sigmaJ = 0.2
    strike = 100.0
    T = 1.0

    model = MertonJump(r, q, sigma, S0, jump_intensity, muJ, sigmaJ, random_state=42)
    n_sims, n_steps = 100_000, 50

    # 1) Delta
    delta_est = pathwise_delta_eurocall(model, strike, T, n_sims, n_steps)
    print(f"[MertonJump] Pathwise Delta: {delta_est:.5f}")

    # 2) Suppose "market_price" is known => find implied vol
    # We'll treat 'sigma' as the model's "sigma" in the drift part. We'll guess it's the same logic.
    # This is somewhat conceptual; for advanced jumps, "vol" might not be a direct param.
    market_price = 12.0
    iv = implied_vol_mc(
        MertonJump,
        r=r,
        q=q,
        S0=S0,
        strike=strike,
        T=T,
        market_price=market_price,
        n_sims=n_sims,
        n_steps=n_steps,
        seed=43,
        vol_low=1e-3,
        vol_high=1.0,
    )
    print(f"[MertonJump] Implied Vol from market_price={market_price} => {iv:.4%}")


if __name__ == "__main__":
    main()
