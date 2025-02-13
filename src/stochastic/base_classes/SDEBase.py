from src.stochastic.StochasticProcess import StochasticProcess
import numpy as np
import matplotlib.pyplot as plt

class SDEBase:
    """
    Represents a single Stochastic Differential Equation.
    """

    def __init__(self, d_process):
        """
        Initialize the SDE with the given differential process.

        Parameters
        ----------
        d_process : StochasticProcess
            Represents the combined differential terms of the SDE.
        """
        if not isinstance(d_process, StochasticProcess):
            raise TypeError("d_process must be an instance of StochasticProcess.")
        self.d_process = d_process

    def sample_paths(self, sims, steps, start=0, end=1):
        """
        Sample paths for the SDE over the specified index range.

        Parameters
        ----------
        sims  : int : Number of sample paths.
        steps : int : Number of time steps.
        start : float : Start of the index range.
        end   : float : End of the index range.

        Returns
        -------
        paths : ndarray : Array of sampled paths.
        """
        times = np.linspace(start, end, steps)
        paths = np.zeros((sims, steps))
        paths[:, 0] = 0  # Assuming the process starts at zero

        for i, t in enumerate(times[1:], start=1):
            delta = self.d_process.sample(sims=sims, idx=t)
            paths[:, i] = paths[:, i - 1] + delta

        return times, paths

    def graph_paths(self, sims, steps, start=0, end=1):
        """
        Plot sample paths for the SDE.

        Parameters
        ----------
        sims  : int : Number of sample paths.
        steps : int : Number of time steps.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        times, paths = self.sample_paths(sims=sims, steps=steps, start=start, end=end)
        plt.figure(figsize=(10, 6))
        for sim in range(sims):
            plt.plot(times, paths[sim, :], label=f"Path {sim + 1}" if sims < 5 else "")
        plt.title(f"Sample Paths of {self.__class__.__name__}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        if sims < 5:
            plt.legend()
        plt.grid()
        plt.show()
