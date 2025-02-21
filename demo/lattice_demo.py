"""
lattice_demo.py
===============

This demo prices 6 options (3 call, 3 put) using two lattice models:
    • CRR lattice (European style, is_american=False)
    • TOPM lattice (European style, is_american=False)

For each (strike, maturity) pair in:
    [(90, 1.0), (100, 0.75), (110, 0.5)]
the demo computes both a call and a put.

It then prints two tables – one for calls and one for puts – where each table
has two rows per option (first row: CRR results, second row: TOPM results).
All numerical values are rounded to 3 decimal places.
"""

import concurrent.futures
from src.techniques.lattice.crr_lattice import CRRLattice
from src.techniques.lattice.topm_lattice import TOPMLattice
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment


def compute_option_results(params):
    """
    Compute pricing and Greeks for a single option (both call and put)
    for a given strike (K) and maturity (T) pair using two lattice methods.
    This function creates its own underlying and market environment objects.
    
    Returns a tuple:
        (option_desc, call_crr, call_topm, put_crr, put_topm)
    where each "xxx" is a tuple: (price, greeks_dict).
    """
    K, T = params
    option_desc = f"Strike={K}, T={T}"

    # Create underlying and market environment (each process builds its own)
    underlying = Stock(spot=100.0, volatility=0.2, dividend=0.02)
    market_env = MarketEnvironment(rate=0.05)

    # ----------------------
    # Compute Call Option Values
    # ----------------------
    # CRR (European)
    crr_call = CRRLattice(
        S0=underlying.spot,
        K=K,
        r=market_env.rate,
        sigma=underlying.volatility,
        q=underlying.dividend,
        T=T,
        steps=100,
        is_call=True,
        is_american=False
    )
    crr_call_price = float(crr_call.price_option())
    crr_call_greeks = crr_call.calc_greeks()

    # TOPM
    topm_call = TOPMLattice(
        S0=underlying.spot,
        K=K,
        r=market_env.rate,
        sigma=underlying.volatility,
        q=underlying.dividend,
        T=T,
        steps=100,
        is_call=True,
        is_american=False
    )
    topm_call_price = float(topm_call.price_option())
    topm_call_greeks = topm_call.calc_greeks()

    # ----------------------
    # Compute Put Option Values
    # ----------------------
    # CRR (European)
    crr_put = CRRLattice(
        S0=underlying.spot,
        K=K,
        r=market_env.rate,
        sigma=underlying.volatility,
        q=underlying.dividend,
        T=T,
        steps=100,
        is_call=False,
        is_american=False
    )
    crr_put_price = float(crr_put.price_option())
    crr_put_greeks = crr_put.calc_greeks()

    topm_put = TOPMLattice(
        S0=underlying.spot,
        K=K,
        r=market_env.rate,
        sigma=underlying.volatility,
        q=underlying.dividend,
        T=T,
        steps=100,
        is_call=False,
        is_american=False
    )
    topm_put_price = float(topm_put.price_option())
    topm_put_greeks = topm_put.calc_greeks()

    return (option_desc,
            (crr_call_price, crr_call_greeks), (topm_call_price, topm_call_greeks),
            (crr_put_price, crr_put_greeks), (topm_put_price, topm_put_greeks))


def print_table(option_type, results):
    """
    Print a formatted table for either Call or Put options.
    
    Parameters:
      - option_type: A string ("Call" or "Put")
      - results: A list of tuples. Each tuple is of the form:
            (option_desc, crr_data, topm_data)
        where crr_data and topm_data are tuples: (price, greeks_dict)
    The table prints two rows per option (CRR first, then TOPM) with a separator line.
    """
    title = f"=== {option_type} Options ==="
    print(title)
    header = f"{'Option':<20} {'Method':<6} {'Price':>10} {'Delta':>10} {'Gamma':>10} {'Theta':>10}"
    print(header)
    for (option_desc, method_crr, method_topm) in results:
        # Unpack CRR values
        price, greeks = method_crr
        delta = greeks['Delta']
        gamma = greeks['Gamma']
        theta = greeks['Theta']
        # Print CRR row (option description shown on first row)
        print(f"{option_desc:<20} {'CRR':<6} {price:10.3f} {delta:10.3f} {gamma:10.3f} {theta:10.3f}")
        # Unpack TOPM values
        price, greeks = method_topm
        delta = greeks['Delta']
        gamma = greeks['Gamma']
        theta = greeks['Theta']
        # Print TOPM row (blank for option column)
        print(f"{'':<20} {'TOPM':<6} {price:10.3f} {delta:10.3f} {gamma:10.3f} {theta:10.3f}")
        print("-" * 60)
    print("\n")


def main():
    # Define option parameters: list of (strike, maturity) pairs.
    option_parameters = [(90, 1.0), (100, 0.75), (110, 0.5)]

    # Prepare lists to hold results for call and put options.
    call_results = []
    put_results = []

    # Use a process pool to compute option results concurrently.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        computed = list(executor.map(compute_option_results, option_parameters))

    # Each computed result is a tuple:
    # (option_desc, (crr_call_price, crr_call_greeks), (topm_call_price, topm_call_greeks),
    #  (crr_put_price, crr_put_greeks), (topm_put_price, topm_put_greeks))
    for res in computed:
        option_desc, call_crr, call_topm, put_crr, put_topm = res
        call_results.append((option_desc, call_crr, call_topm))
        put_results.append((option_desc, put_crr, put_topm))

    # Print formatted tables for calls and puts.
    print_table("Call", call_results)
    print_table("Put", put_results)


if __name__ == "__main__":
    main()
