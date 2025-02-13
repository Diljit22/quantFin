import numpy as np
from src.stochastic.GammaProcess import GammaProcess
from src.stochastic.BrownianMotion import BrownianMotion
from src.stochastic.StochasticProcess import StochasticProcess

class VarianceGammaProcess(StochasticProcess):
    """
    Variance Gamma Process combining a Gamma Process and a Brownian Motion.
    The VG process is constructed as:
        X_t = drift * t + magnitude * (BM_{Gamma_t}),
    where Gamma_t is the Gamma process and BM is a standard Brownian motion.
    """

    def __init__(self, magBM=1, magGamma=1, theta=0, lam=1, start=0, end=1):
        """
        Initialize the Variance Gamma Process parameters.

        Parameters
        ----------
        magBM : float : Magnitude of the Brownian Motion component.
        magGamma : float : Magnitude of the Gamma process.
        theta : float : Mean of the Gamma process increments.
        lam : float : Lambda (rate parameter) of the Gamma process.
        start : float : Start of the time index.
        end   : float : End of the time index.
        """
        super().__init__(start=start, end=end)
        self.Gamma = GammaProcess(theta=theta, lam=lam, mag=magGamma, start=start, end=end)
        self.Brownian = BrownianMotion(mag=magBM, start=start, end=end)

    def sample(self, sims, idx, shape=None):
        """
        Sample from the Variance Gamma process.

        Parameters
        ----------
        sims : int : Number of simulations.
        idx  : float : Time point at which the process is sampled.
        shape : tuple, optional : Desired output shape.

        Returns
        -------
        np.ndarray : Samples from the Variance Gamma process.
        """
        if shape is None:
            shape =(sims,)

        gamma_samples = self.Gamma.sample(sims, idx, shape)
        bm_samples = self.Brownian.sample(sims, gamma_samples, shape)
        return bm_samples

    def graph(self, numPaths=1, steps=100, showComponents=False):
        """
        Plot sample paths of the Variance Gamma process.

        Parameters
        ----------
        numPaths : int : Number of paths to generate.
        steps : int : Number of time steps in the path.
        showComponents : bool : If True, plot the Gamma and Brownian components separately.
        """
        indexSet = np.linspace(self.index[0], self.index[1], steps)
        vg_paths = np.zeros((numPaths, len(indexSet)))

        for i, t in enumerate(indexSet):
            samples = self.sample(sims=numPaths, idx=t)
            vg_paths[:, i] = samples

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        if showComponents:
            gamma_paths = np.zeros((numPaths, len(indexSet)))
            bm_paths = np.zeros((numPaths, len(indexSet)))
            for i, t in enumerate(indexSet):
                gamma_samples = self.Gamma.sample(sims=numPaths, idx=t)
                gamma_paths[:, i] = gamma_samples
                bm_samples = self.Brownian.sample(sims=numPaths, idx=gamma_samples)
                bm_paths[:, i] = bm_samples

            for path in gamma_paths:
                plt.plot(indexSet, path, label="Gamma Process", linestyle="--", alpha=0.6)
            for path in bm_paths:
                plt.plot(indexSet, path, label="Brownian Motion", linestyle=":", alpha=0.6)

        for path in vg_paths:
            plt.plot(indexSet, path, label="Variance Gamma Process", alpha=0.8)

        plt.title(f"Sample Paths of {self.__class__.__name__}")
        plt.xlabel("Index (t)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

    def __repr__(self):
        """
        Represent the Variance Gamma Process.
        """
        params = (f"magBM={self.Brownian.mag}, "
                  f"magGamma={self.Gamma.mag}, theta={self.Gamma.theta}, "
                  f"lam={self.Gamma.lam}")
        return f"{self.__class__.__name__}({params})"

if __name__ == "__main__":
    # Example usage
    vg = VarianceGammaProcess(magBM=0.2, magGamma=0.1, theta=0.05, lam=0.2)
    vg.graph(numPaths=5, showComponents=True)
