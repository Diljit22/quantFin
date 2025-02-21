"""
sde_models.py
=============

Concrete classes for a variety of SDE-based models:

1) BlackScholesMerton
2) MertonJump
3) DupireLocalVol
4) CEV
5) Heston
6) KouJump
7) Bates (Heston + jumps)
8) SABR
9) VarianceGamma
10) CGMY
11) NIG
12) OrnsteinUhlenbeck

Each class implements sample_paths(...) in a vectorized manner using NumPy.
"""

import numpy as np
import math
from typing import Optional, Callable

from src.stochastic.base_sde import BaseSDEModel


###############################################################################
# 1) Black–Scholes–Merton
###############################################################################
class BlackScholesMerton(BaseSDEModel):
    """
    Standard GBM: dS = S*(r - q)*dt + S*sigma*dW.

    Example
    -------
    >>> model = BlackScholesMerton(r=0.05, q=0.01, sigma=0.2, S0=100.0)
    >>> paths = model.sample_paths(T=1.0, n_sims=10000, n_steps=50)
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
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        self.sigma = sigma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            log_s = np.log(paths[:, step])
            log_s_next = log_s + drift + vol * Z
            paths[:, step + 1] = np.exp(log_s_next)

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}"


###############################################################################
# 2) Merton Jump-Diffusion
###############################################################################
class MertonJump(BaseSDEModel):
    """
    Merton jump model: dS = S[(r-q-lambda*(m_j-1))-0.5*sigma^2] dt + S*sigma dW + jumps.

    Jumps: Poisson(λ dt). Each jump ~ lognormal( muJ, sigmaJ ).
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        S0: float,
        jump_intensity: float,
        muJ: float,
        sigmaJ: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if jump_intensity < 0:
            raise ValueError("jump_intensity must be >= 0.")
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        # E[J] = exp(muJ + 0.5*sigmaJ^2)
        self.m_j = math.exp(muJ + 0.5 * sigmaJ**2)

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0

        drift_correction = (self.r - self.q) - self.jump_intensity * (self.m_j - 1)
        drift = (drift_correction - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            log_s = np.log(paths[:, step])
            log_s_next = log_s + drift + vol * Z
            S_next = np.exp(log_s_next)
            # Poisson jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    # product of jump_i
                    # each jump_i = exp( normal(muJ, sigmaJ) )
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_jumps[i])
                    J_factor = np.exp(Y).prod()
                    S_next[i] *= J_factor
            paths[:, step + 1] = S_next

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}, sigma={self.sigma}, jump_intensity={self.jump_intensity}, "
            f"muJ={self.muJ}, sigmaJ={self.sigmaJ}"
        )


###############################################################################
# 3) Dupire Local Vol
###############################################################################
class DupireLocalVol(BaseSDEModel):
    """
    Local volatility model with σ = sigma_fn(S,t).
    Usually derived from implied vol surface (Dupire's formula).

    sigma_fn : Callable[[float, float], float]
        A function sigma_fn(S, t) -> float giving local volatility.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        sigma_fn: Callable[[float, float], float],
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.sigma_fn = sigma_fn

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0

        for step in range(n_steps):
            t_cur = step * dt
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            S_cur = paths[:, step]
            # Vectorized local vol for each path
            sigmas = np.array([self.sigma_fn(s, t_cur) for s in S_cur])
            drift = (self.r - self.q) * dt
            # We do basic Euler on log(S):
            log_s = np.log(S_cur + 1e-14)  # avoid zero
            log_s_next = (
                log_s + (drift - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * Z
            )
            paths[:, step + 1] = np.exp(log_s_next)

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, DupireLocalVol(sigma_fn=...)"


###############################################################################
# 4) CEV (Constant Elasticity of Variance)
###############################################################################
class CEV(BaseSDEModel):
    """
    dS = alpha*S*dt + beta*S^gamma * dW, or simplified to (r-q) plus the CEV term.

    We'll do: dS = S*(r - q)*dt + c*(S^gamma)* dW.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        c: float,
        gamma: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        if c <= 0:
            raise ValueError("CEV c must be > 0.")
        self.c = c
        self.gamma = gamma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        drift = self.r - self.q

        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            S_cur = paths[:, step]
            # Euler on S:
            S_next = (
                S_cur
                + drift * S_cur * dt
                + self.c * np.power(S_cur, self.gamma) * np.sqrt(dt) * Z
            )
            # no negativity clamp
            S_next = np.clip(S_next, 1e-14, None)
            paths[:, step + 1] = S_next

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, CEV(c={self.c}, gamma={self.gamma})"


###############################################################################
# 5) Heston
###############################################################################
class Heston(BaseSDEModel):
    """
    2D system: dv = kappa*(theta - v)*dt + sigma_v sqrt(v)* dWv
               dS = S*(r-q)*dt + S* sqrt(v)* dWs,
    Corr(dWv,dWs)=rho
    We'll store only S in the result, not the v path.

    For positivity, we clamp v=0 if negative after Euler.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        if abs(rho) > 1:
            raise ValueError("Correlation rho must be in [-1,1].")

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0

        v_arr = np.full(n_sims, self.v0, dtype=float)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # Update v
            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, a_min=0, a_max=None)

            # Update S
            s_old = paths[:, step]
            drift = (self.r - self.q) * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s
            paths[:, step + 1] = s_old * np.exp(drift + diff)

            v_arr = v_next
        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, Heston(v0={self.v0}, kappa={self.kappa}, theta={self.theta}, sigma_v={self.sigma_v}, rho={self.rho})"


###############################################################################
# 6) Kou Jump
###############################################################################
class KouJump(BaseSDEModel):
    """
    Kou's double-exponential jump-diffusion. Poisson(λ dt).
    Jump size distribution: mixture of exponentials for upward/downward jumps.

    For simplicity, we store p_up, eta1, eta2 as in Kou's model.
    dS = S[(r-q - λ * (E[J] -1)) - 0.5*sigma^2]*dt + S*sigma*dW + jumps

    E[J] = p_up*(eta1/(eta1-1)) + (1-p_up)*(eta2/(eta2+1)).
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        sigma: float,
        lambda_: float,
        p_up: float,
        eta1: float,
        eta2: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma
        self.lambda_ = lambda_
        self.p_up = p_up
        self.eta1 = eta1
        self.eta2 = eta2
        # E[J], from known double-exponential distribution
        # If X>0 upward jump, X<0 downward jump => We store J= e^X
        # E[J] = p_up * (eta1/(eta1-1)) + (1-p_up)*(eta2/(eta2+1)) for 1<eta1, eta2>0
        EJ_up = eta1 / (eta1 - 1.0) if eta1 > 1 else math.inf
        EJ_dn = eta2 / (eta2 + 1.0) if eta2 > 0 else math.inf
        self.ej = p_up * EJ_up + (1 - p_up) * EJ_dn

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = self.S0
        drift_correction = (self.r - self.q) - self.lambda_ * (self.ej - 1)
        drift = lambda step: (drift_correction - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s = np.log(paths[:, step])
            log_s_next = log_s + drift(step) + vol * Z
            S_next = np.exp(log_s_next)
            # Poisson jumps
            N_j = self._rng.poisson(self.lambda_ * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    # for each jump, pick upward or downward
                    jumps = 1.0
                    Y_updown = self._rng.uniform(size=N_j[i])
                    for yv in Y_updown:
                        if yv < self.p_up:
                            # upward jump => X ~ Exp(eta1)
                            x = self._rng.exponential(1.0 / self.eta1)
                            jumps *= math.exp(x)
                        else:
                            # downward => -Exp(eta2)
                            x = self._rng.exponential(1.0 / self.eta2)
                            jumps *= math.exp(-x)
                    S_next[i] *= jumps
            paths[:, step + 1] = S_next

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}, KouJump(sigma={self.sigma}, lambda={self.lambda_}, p_up={self.p_up}, "
            f"eta1={self.eta1}, eta2={self.eta2})"
        )


###############################################################################
# 7) Bates (Heston + jumps)
###############################################################################
class Bates(BaseSDEModel):
    """
    Combine Heston with a Poisson jump process (lognormal or double-exp).
    We'll do lognormal jumps for brevity. Similar to Merton jump but integrated in Heston's vol.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        jump_intensity: float,
        muJ: float,
        sigmaJ: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.jump_intensity = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.EJ = math.exp(muJ + 0.5 * sigmaJ**2)

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = self.S0
        v_arr = np.full(n_sims, self.v0, dtype=float)
        sqrt_dt = math.sqrt(dt)

        # drift correction
        drift_correction = (self.r - self.q) - self.jump_intensity * (self.EJ - 1)

        for step in range(n_steps):
            # correlated normals
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            Z_v = Z1
            Z_s = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # update v
            v_next = (
                v_arr
                + self.kappa * (self.theta - v_arr) * dt
                + self.sigma_v * np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_v
            )
            v_next = np.clip(v_next, 0, None)

            # update S ignoring jump
            s_old = paths[:, step]
            drift = drift_correction * dt - 0.5 * v_arr * dt
            diff = np.sqrt(np.maximum(v_arr, 0)) * sqrt_dt * Z_s
            s_temp = s_old * np.exp(drift + diff)

            # jump
            N_j = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_j[i] > 0:
                    # multiply by product of lognormal jumps
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_j[i])
                    s_temp[i] *= np.exp(Y.sum())

            paths[:, step + 1] = s_temp
            v_arr = v_next

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}, Bates(Heston params..., jump_intensity={self.jump_intensity}, "
            f"muJ={self.muJ}, sigmaJ={self.sigmaJ})"
        )


###############################################################################
# 8) SABR
###############################################################################
class SABR(BaseSDEModel):
    """
    A minimal 2D SABR approach for F_t, alpha_t. For equity we do S ~ ?

    We'll treat S as F. The drift is typically 0 for forward, but let's do (r-q).
    dF = alpha * F^beta dW_s
    dalpha = nu alpha dW_alpha
    Corr(dW_s, dW_alpha)=rho
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        alpha0: float,
        beta: float,
        nu: float,
        rho: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.alpha0 = alpha0
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = self.S0
        alpha_arr = np.full(n_sims, self.alpha0)
        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            Z_s = Z1
            Z_a = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # update alpha
            alpha_next = alpha_arr + self.nu * alpha_arr * sqrt_dt * Z_a
            alpha_next = np.clip(alpha_next, 1e-14, None)

            # update F (S in eq)
            s_old = paths[:, step]
            # standard SABR has no drift for forward, but let's do r-q if we want
            drift = (self.r - self.q) * dt
            vol_term = (
                alpha_arr
                * np.power(np.clip(s_old, 1e-14, None), self.beta)
                * sqrt_dt
                * Z_s
            )
            s_new = s_old * np.exp(
                drift - 0.5 * (alpha_arr**2) * dt
            )  # might not be correct ...
            # but let's do direct Euler:
            s_new = s_old + drift * s_old + vol_term

            paths[:, step + 1] = np.clip(s_new, 1e-14, None)
            alpha_arr = alpha_next

        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, SABR(alpha0={self.alpha0}, beta={self.beta}, nu={self.nu}, rho={self.rho})"


###############################################################################
# 9) Variance Gamma
###############################################################################
class VarianceGamma(BaseSDEModel):
    """
    A pure-jump model: S_t = S0 + drift * G_t + vol * W_{G_t},
    where G_t ~ Gamma process. We'll do a simplified approach (Madan-Seneta).
    The path discretization uses subordination in gamma steps, etc.
    Real code might require more advanced approach or direct varianceGamma() sampling.

    We'll just do a rough Euler for demonstration.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        sigma: float,
        theta: float,
        nu: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.sigma = sigma
        self.theta = theta
        self.nu = nu

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        # might do partial approach for subordinator
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        mu = self.r - self.q

        for step in range(n_steps):
            # draw gamma increments G ~ Gamma(dt/nu, nu) ?
            # shape=dt/nu, scale=nu
            G = self._rng.gamma(shape=dt / self.nu, scale=self.nu, size=n_sims)
            # Then stock increment: dX = theta*G + sigma* sqrt(G)*Z
            Z = self._rng.normal(size=n_sims)
            dX = self.theta * G + self.sigma * np.sqrt(G) * Z
            S_old = paths[:, step]
            paths[:, step + 1] = S_old * np.exp((mu) * dt + dX)  # simplistic
        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, VarianceGamma(sigma={self.sigma}, theta={self.theta}, nu={self.nu})"


###############################################################################
# 10) CGMY
###############################################################################
class CGMY(BaseSDEModel):
    """
    Another Lévy model with infinite activity. We'll do a simplistic Euler approach.
    Real-world usage often uses specialized FFT or measure-based discretization.

    This is a skeleton to illustrate the structure. Implementation is partial.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        C: float,
        G: float,
        M: float,
        Y: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        # In practice, you'd do a small-jump + large-jump decomposition or an FFT approach
        # Here we just place a placeholder
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        for step in range(n_steps):
            # placeholder random increments
            # a naive approach might do compound Poisson for big jumps + stable for small jumps...
            # We'll do a "dummy zero" increment
            dX = np.zeros(n_sims)
            S_old = paths[:, step]
            paths[:, step + 1] = S_old * np.exp((self.r - self.q) * dt + dX)
        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, CGMY(C={self.C}, G={self.G}, M={self.M}, Y={self.Y})"


###############################################################################
# 11) NIG (Normal Inverse Gaussian)
###############################################################################
class NIG(BaseSDEModel):
    """
    Another pure-jump model. Typically used for advanced equity or FX skew.
    This is a placeholder: real code might do subordination or advanced bridging.
    """

    def __init__(
        self,
        r: float,
        q: float,
        S0: float,
        alpha: float,
        beta: float,
        delta: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        for step in range(n_steps):
            # placeholder increments
            S_old = paths[:, step]
            dX = np.zeros(n_sims)  # a real NIG approach would do a subordinator
            paths[:, step + 1] = S_old * np.exp((self.r - self.q) * dt + dX)
        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, NIG(alpha={self.alpha}, beta={self.beta}, delta={self.delta})"


###############################################################################
# 12) Ornstein–Uhlenbeck (If relevant for rates/mean reversion)
###############################################################################
class OrnsteinUhlenbeck(BaseSDEModel):
    """
    If you want to treat S as an OU process, typically for interest rates or
    commodity convenience yield. For option pricing on an OU under positive transforms, etc.
    dX = theta*(mu - X_t)*dt + sigma dW.
    We'll interpret 'S' as X, so watch negativity if you transform to price.
    """

    def __init__(
        self,
        r: float,
        q: float,
        X0: float,
        theta: float,
        mu: float,
        sigma: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, X0, random_state)  # S0 is "X0"
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = self.S0
        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            X_cur = paths[:, step]
            X_next = (
                X_cur
                + self.theta * (self.mu - X_cur) * dt
                + self.sigma * np.sqrt(dt) * Z
            )
            paths[:, step + 1] = X_next
        return paths

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, OrnsteinUhlenbeck(theta={self.theta}, mu={self.mu}, sigma={self.sigma})"
