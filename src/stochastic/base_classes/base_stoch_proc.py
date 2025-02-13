#!/usr/bin/env python3
# stochastic_process_base.py

"""
StochasticProcess Base Class (Top-Tier HPC-Friendly)
====================================================

Provides a foundation for creating, combining, and sampling from
stochastic processes.

Key Features
------------
1. Abstract base class `StochasticProcess`: 
   - Enforces a `sample(...)` interface for a given time index.
   - Provides a default `simulate_paths(...)` method for multi-step
     sampling across a time grid, with optional concurrency.

2. HPC & Concurrency:
   - If `use_parallel=True`, a chunk-based approach can distribute large
     Monte Carlo tasks across threads/processes.

   - Allows combining processes (add/sub) and applying functions pointwise.

4. Extensibility:
   - Derive new child classes for specific processes (e.g. BrownianMotion,
     OrnsteinUhlenbeck, Heston, VarianceGamma, etc.).
   - Implement specialized logic or override `simulate_paths` for advanced SDE 
     schemes or HPC methods.

Usage Example
-------------
>>> import numpy as np
>>> from stochastic_process_base import StochasticProcess

>>> # Suppose we define a child class MyProcess(StochasticProcess):
>>> #  - implement sample() for single times, graph() if desired
>>> # Then:
>>> p = MyProcess(start=0.0, end=1.0)
>>> final_samples = p.sample(sims=10000, idx=1.0)
>>> # HPC path simulation:
>>> time_grid = np.linspace(0,1,100)
>>> all_paths = p.simulate_paths(n_paths=5000, time_grid=time_grid, use_parallel=True)
>>> print("all_paths shape =", all_paths.shape)  # (5000, 100)
"""

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Callable
import concurrent.futures


class StochasticProcess(ABC):
    """
    Base abstract class for stochastic processes.

    Provides:
      - A foundation for sampling the process at a given index/time (sample()).
      - A default HPC-friendly method simulate_paths() for multi-step path simulation
        across a time grid, which child classes may override for specialized SDE logic.
      - Basic addition, subtraction, scalar multiplication, and function application,
        returning new combined or transformed processes.

    For HPC concurrency:
      - If `use_parallel=True` in simulate_paths, chunk the path computations
        across multiple threads or processes. For advanced usage, consider joblib
        or GPU-based approaches if required.

    Notes
    -----
    - Indices [start, end] define the domain of the process (time or other parameter).
    - Child classes must implement sample(...) at minimum.
    """

    def __init__(self, start: float = 0.0, end: float = 1.0) -> None:
        """
        Initialize the stochastic process domain.

        Parameters
        ----------
        start : float, optional
            The start of the index range (e.g. t=0).
        end : float, optional
            The end of the index range (e.g. t=1).
        """
        self.index = [start, end]

    @abstractmethod
    def sample(
        self, sims: int, idx: float, shape: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Sample the process at a given index/time.

        Parameters
        ----------
        sims : int
            Number of independent samples to draw.
        idx : float
            The time or index in [self.index[0], self.index[1]].
        shape : int or tuple of int, optional
            Desired output shape. If None, typically returns a 1D array
            of length `sims`. If shape is an int or a tuple, reshape
            the samples accordingly.

        Returns
        -------
        np.ndarray
            Samples from the process at time idx.
        """
        raise NotImplementedError

    @abstractmethod
    def graph(self, num_paths: int = 1, steps: int = 100) -> None:
        """
        Plot sample paths of the process

        Parameters
        ----------
        num_paths : int, optional
            How many paths to plot.
        steps : int, optional
            Number of discrete steps in [start, end] for the plot.
        """
        raise NotImplementedError

    def simulate_paths(
        self,
        n_paths: int,
        time_grid: np.ndarray,
        use_parallel: bool = False,
        chunk_size: int = 0,
    ) -> np.ndarray:
        """
        Simulate multiple sample paths over the specified time_grid, using
        repeated calls to `sample(...)` at each time step.

        This default implementation is HPC-friendly but:
          - For each time in time_grid, call sample(n_paths, t), storing results.
          - This is feasible if `sample(...)` itself is vectorized or HPC-optimized
            in the child class.
          - For large n_paths or advanced SDE logic, a specialized approach
            (Euler, Milstein, etc.) might be faster, so a child can override.

        If `use_parallel=True`, chunk the n_paths across threads. This concurrency
        can help if `sample(...)` is CPU-bound. For truly large path+time combos,
        consider joblib or GPU approaches.

        Parameters
        ----------
        n_paths : int
            Number of independent paths to simulate.
        time_grid : np.ndarray
            Array of times in ascending order. shape=(n_times,).
        use_parallel : bool, optional
            If True, distribute path computations across multiple threads.
        chunk_size : int, optional
            If >0, manual chunk size for dividing n_paths among threads.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_paths, n_times) containing the sample
            paths at each time in time_grid.

        Notes
        -----
        1. Thread-safety: if sample(...) modifies shared state, concurrency
           might cause issues. Typically sample(...) is pure, so it's safe.
        2. For large HPC, process-based parallel or GPU might be preferred.
        3. This default method does not do time stepping with correlation across
           steps. It simply treats each time as independent calls to sample(...).
           That might be incorrect for many SDE-based processes, so child classes
           must override this entire method with a more consistent scheme.
        """
        times = np.asarray(time_grid)
        n_times = len(times)
        paths = np.zeros((n_paths, n_times), dtype=float)

        def worker_block(start_idx: int, end_idx: int):
            path_count = end_idx - start_idx
            for i_t, t_val in enumerate(times):
                block_samples = self.sample(path_count, t_val)  # shape=(path_count,)
                paths[start_idx:end_idx, i_t] = block_samples

        if not use_parallel or n_paths <= 1:
            # single-thread
            worker_block(0, n_paths)
        else:
            # chunk approach for concurrency
            if chunk_size <= 0:
                chunk_size = max(256, n_paths // 4)
            tasks = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                start = 0
                while start < n_paths:
                    end = min(start + chunk_size, n_paths)
                    tasks.append(executor.submit(worker_block, start, end))
                    start = end
                concurrent.futures.wait(tasks)

        return paths


    @classmethod
    def create_deterministic(
        cls, value: float, start: float, end: float
    ) -> "StochasticProcess":
        """
        Creates a deterministic process representing a constant value over time.

        Parameters
        ----------
        value : float
            The constant value of the process.
        start : float
            Start of the index range.
        end : float
            End of the index range.

        Returns
        -------
        StochasticProcess
            A deterministic process that always yields 'value'.
        """

        class DeterministicProcess(cls):
            def sample(
                self,
                sims: int,
                idx: float,
                shape: Optional[Union[int, Tuple[int, ...]]] = None,
            ) -> np.ndarray:
                if shape is None:
                    shape = (sims,)
                return np.full(shape, value, dtype=float)

            def graph(self, num_paths: int = 1, steps: int = 100) -> None:
                t_array = np.linspace(start, end, steps)
                results = np.full((num_paths, steps), value, dtype=float)

                plt.figure(figsize=(10, 6))
                for p_idx in range(num_paths):
                    plt.plot(
                        t_array, results[p_idx, :], label=f"Deterministic {p_idx+1}"
                    )
                plt.title("Deterministic Process")
                plt.xlabel("Index (t)")
                plt.ylabel("Value")
                plt.grid()
                plt.legend()
                plt.show()

        return DeterministicProcess(start=start, end=end)

    def __add__(self, other: Union[float, "StochasticProcess"]) -> "StochasticProcess":
        """
        Add another process or a scalar to this process.

        Parameters
        ----------
        other : float or StochasticProcess
            The value to add.

        Returns
        -------
        StochasticProcess
            A new process representing this + other.
        """
        if isinstance(other, StochasticProcess):
            return CombinedProcess(self, other, operation="+")
        elif isinstance(other, (int, float)):
            det = self.create_deterministic(float(other), self.index[0], self.index[1])
            return self + det
        else:
            raise ValueError("Unsupported type for addition.")

    def __radd__(self, other: Union[float, "StochasticProcess"]) -> "StochasticProcess":
        """Scalar + self => calls self.__add__."""
        return self.__add__(other)

    def __sub__(self, other: Union[float, "StochasticProcess"]) -> "StochasticProcess":
        """
        Subtract another process or scalar from this process.
        """
        if isinstance(other, StochasticProcess):
            return CombinedProcess(self, other, operation="-")
        elif isinstance(other, (int, float)):
            det = self.create_deterministic(float(other), self.index[0], self.index[1])
            # self - det
            return CombinedProcess(self, det, operation="-")
        else:
            raise ValueError("Unsupported type for subtraction.")

    def __rsub__(self, other: Union[float, "StochasticProcess"]) -> "StochasticProcess":
        """
        other - self
        """
        if isinstance(other, (int, float)):
            det = self.create_deterministic(float(other), self.index[0], self.index[1])
            return CombinedProcess(det, self, operation="-")
        elif isinstance(other, StochasticProcess):
            return CombinedProcess(other, self, operation="-")
        else:
            raise ValueError("Unsupported type for subtraction.")

    def __mul__(self, scalar: float) -> "StochasticProcess":
        """
        Scalar multiplication: process * scalar.
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError("Multiplication only supports numeric scalars.")
        return CombinedProcess(self, None, operation="*", scalar=float(scalar))

    def __rmul__(self, scalar: float) -> "StochasticProcess":
        """scalar * process => same as __mul__."""
        return self.__mul__(scalar)

    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> "StochasticProcess":
        """
        Apply a numpy-compatible function pointwise to the samples.

        Parameters
        ----------
        func : callable
            e.g. np.exp, np.sqrt, etc.

        Returns
        -------
        StochasticProcess
            A new wrapped process representing func(self).
        """
        return FunctionAppliedProcess(self, func)

class CombinedProcess(StochasticProcess):
    """
    Represents a combination of two processes or a scalar operation.
    E.g., for 'operation' in {+, -, *}.

    If 'scalar' is not None, do scalar op. Otherwise, we combine process1 and process2.
    """

    def __init__(
        self,
        process1: StochasticProcess,
        process2: Optional[StochasticProcess] = None,
        operation: str = "+",
        scalar: Optional[float] = None,
    ):
        if scalar is not None and process2 is None:
            new_index = process1.index
        elif process2 is not None:
            new_index = [
                max(process1.index[0], process2.index[0]),
                min(process1.index[1], process2.index[1]),
            ]
        else:
            raise ValueError("Either process2 or scalar must be specified.")

        super().__init__(start=new_index[0], end=new_index[1])
        self.process1 = process1
        self.process2 = process2
        self.operation = operation
        self.scalar = scalar

    def sample(
        self, sims: int, idx: float, shape: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Sample the combined or scalar-modified process.

        E.g., if operation='+' and process2 is not None, do p1 + p2.
              if operation='*' and scalar is not None, do scalar * p1.
        """
        samp1 = self.process1.sample(sims, idx)
        if shape is None:
            shape = (sims,)

        if self.scalar is not None:
            if self.operation == "*":
                out = self.scalar * samp1
            elif self.operation == "+":
                out = self.scalar + samp1
            elif self.operation == "-":
                out = samp1 - self.scalar
            else:
                raise ValueError(f"Unsupported scalar operation {self.operation}")
            return out.reshape(shape)
        elif self.process2 is not None:
            samp2 = self.process2.sample(sims, idx)
            if self.operation == "+":
                out = samp1 + samp2
            elif self.operation == "-":
                out = samp1 - samp2
            elif self.operation == "*":
                out = samp1 * samp2
            else:
                raise ValueError(f"Unsupported process operation {self.operation}")
            return out.reshape(shape)
        else:
            raise ValueError("Invalid CombinedProcess usage.")

    def graph(self, num_paths: int = 1, steps: int = 100) -> None:
        """
        By default, we rely on the parent's graph logic, which calls
        sample(...) at each time. That will combine or do scalar ops
        on child processes.
        """
        super().graph(num_paths, steps)

    def __repr__(self) -> str:
        if self.scalar is not None:
            return f"CombinedProcess({self.process1!r} {self.operation} {self.scalar})"
        else:
            return (
                f"CombinedProcess({self.process1!r} {self.operation} {self.process2!r})"
            )


class FunctionAppliedProcess(StochasticProcess):
    """
    A stochastic process that applies a function (e.g. np.exp) to
    the samples of another process pointwise.
    """

    def __init__(
        self, base_proc: StochasticProcess, func: Callable[[np.ndarray], np.ndarray]
    ):
        start, end = base_proc.index
        super().__init__(start, end)
        self.base_proc = base_proc
        self.func = func

    def sample(
        self, sims: int, idx: float, shape: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        vals = self.base_proc.sample(sims, idx)
        out = self.func(vals)
        if shape is None:
            return out
        else:
            return out.reshape(shape)

    def graph(self, num_paths: int = 1, steps: int = 100) -> None:
        super().graph(num_paths, steps)

    def __repr__(self) -> str:
        return f"FunctionAppliedProcess({self.base_proc!r}, {self.func})"
