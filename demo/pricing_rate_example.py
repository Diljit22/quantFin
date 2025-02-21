###############################################################################
# Example usage
###############################################################################
from src.underlyings.zero_coupon import price_zcb
from src.stochastic.sde.two_factor_gaussian import G2pp
from src.stochastic.sde.vasicek import Vasicek


def main():
    # Example: Vasicek with possible negative rates
    model = Vasicek(r0=0.01, a=0.5, b=0.02, sigma=0.01, random_state=42)
    T = 2.0
    n_sims = 50_000
    n_steps = 50
    zcb_price = price_zcb(model, T, n_sims, n_steps)
    print(f"[Vasicek] ZCB price for T={T:.2f}: {zcb_price:.6f}")

    # G2++ example
    g2model = G2pp(
        r0=0.01,
        a=0.1,
        b=0.2,
        sigma=0.01,
        eta=0.015,
        rho=-0.5,
        phi=0.0,
        random_state=123,
    )
    zcb2 = price_zcb(g2model, T=2.0, n_sims=30_000, n_steps=40)
    print(f"[G2++] ZCB price for T=2.0: {zcb2:.6f}")


if __name__ == "__main__":
    main()
