from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class OptionEvaluation:
    """
    Container for option evaluation results.

    Attributes
    ----------
    model : str
        The name of the pricing model used (e.g., "BlackScholesMerton").
    technique : str
        The evaluation technique used (e.g., "ClosedForm", "FiniteDiff").
    price : float
        The computed option price.
    delta : Optional[float]
        The option delta (optional).
    gamma : Optional[float]
        The option gamma (optional).
    vega : Optional[float]
        The option vega (optional).
    theta : Optional[float]
        The option theta (optional).
    rho : Optional[float]
        The option rho (optional).
    implied_vol : Optional[float]
        The computed implied volatility (optional).
    instrument_data : Optional[Any]
        Additional data describing the option (e.g., strike, maturity, is_call).
    underlying_data : Optional[Any]
        Additional data describing the underlying (e.g., spot, dividend, volatility).
    """

    model: str
    technique: str
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    implied_vol: Optional[float] = None
    instrument_data: Optional[Any] = None
    underlying_data: Optional[Any] = None

    def __str__(self) -> str:
        lines = [
            "Option Evaluation:",
            f"  Model: {self.model}",
            f"  Technique: {self.technique}",
            f"  Price: {self.price:.4f}",
        ]
        if self.delta is not None:
            lines.append(f"  Delta: {self.delta:.4f}")
        if self.gamma is not None:
            lines.append(f"  Gamma: {self.gamma:.4f}")
        if self.vega is not None:
            lines.append(f"  Vega: {self.vega:.4f}")
        if self.theta is not None:
            lines.append(f"  Theta: {self.theta:.4f}")
        if self.rho is not None:
            lines.append(f"  Rho: {self.rho:.4f}")
        if self.implied_vol is not None:
            lines.append(f"  Implied Volatility: {self.implied_vol:.4f}")
        if self.instrument_data is not None:
            lines.append(f"  Instrument Data: {self.instrument_data}")
        if self.underlying_data is not None:
            lines.append(f"  Underlying Data: {self.underlying_data}")
        return "\n".join(lines)
