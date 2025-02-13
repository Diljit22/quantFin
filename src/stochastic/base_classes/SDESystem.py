class SDESystem:
    """
    Represents a system of coupled SDEs.
    """

    def __init__(self, equations, rho=None):
        """
        Initialize the system of SDEs.

        Parameters
        ----------
        equations : list[SDEBase]
            List of SDEs in the system.
        rho : ndarray, optional
            Correlation matrix for the system.
        """
        if not all(isinstance(eq, SDEBase) for eq in equations):
            raise TypeError("All equations must be instances of SDEBase.")
        self.equations = equations
        self.rho = rho

    def sample_paths(self, sims, steps, start=0, end=1):
        """
        Sample paths for the system of SDEs.

        Parameters
        ----------
        sims  : int : Number of sample paths.
        steps : int : Number of time steps.
        start : float : Start of the index range.
        end   : float : End of the index range.

        Returns
        -------
        paths : list[ndarray] : List of arrays containing sampled paths for each SDE.
        """
        num_eqs = len(self.equations)
        times = np.linspace(start, end, steps)
        paths = [np.zeros((sims, steps)) for _ in range(num_eqs)]

        # Correlated increments (if rho is provided)
        correlated_noise = None
        if self.rho is not None:
            correlated_noise = np.linalg.cholesky(self.rho).dot(
                np.random.normal(size=(num_eqs, sims, steps - 1))
            )

        for i, t in enumerate(times[1:], start=1):
            for j, eq in enumerate(self.equations):
                delta = eq.d_process.sample(
                    sims=sims, idx=t
                )
                if correlated_noise is not None:
                    delta += correlated_noise[j, :, i - 1]
                paths[j][:, i] = paths[j][:, i - 1] + delta

        return times, paths

    def graph_paths(self, sims, steps, start=0, end=1):
        """
        Plot sample paths for the system of SDEs.

        Parameters
        ----------
        sims  : int : Number of sample paths.
        steps : int : Number of time steps.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        times, paths = self.sample_paths(sims, steps, start, end)
        plt.figure(figsize=(10, 6))
        for i, path in enumerate(paths):
            for sim in range(sims):
                plt.plot(
                    times, path[sim, :], label=f"SDE {i + 1}, Path {sim + 1}" if sims < 5 else ""
                )
        plt.title("Sample Paths of SDE System")
        plt.xlabel("Time")
        plt.ylabel("Value")
        if sims < 5:
            plt.legend()
        plt.grid()
        plt.show()
