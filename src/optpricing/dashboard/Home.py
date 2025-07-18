import streamlit as st

__doc__ = """
Streamlit home page for the optpricing dashboard.

Shows navigation instructions, available tools and a quick overview
of each page in the app.
"""

st.set_page_config(
    page_title="optpricing | Home",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Welcome to the `optpricing` Library Showcase!")
st.sidebar.success("Select a tool from the sidebar to begin.")
# noqa E501
st.markdown(
    """
    This application is a showcase for a quantitative library.
    It demonstrates workflows from pricing options to calibrating models against
    market data.

    ### How to Use This Dashboard

    1.  **Select a Tool:** Use the sidebar on the left to navigate between the different
        analysis pages.
    2.  **Configure Parameters:** Each page uses a set of controls in the sidebar
        to select the Ticker, Market Data Date, and the primary Model for analysis.
    3.  **Analyze the Results:** Explore the interactive charts, data tables, and
        financial metrics generated by the underlying library.

    ---

    ### Available Tools:

    -   **1_Pricer_and_Greeks:** An interactive page to price an option with any
        supported model and technique combination.
        See the effect of changing parameters in real-time.

    -   **2_Calibration_and_IV:** A model calibration lab. Calibrate models to market
        data and visualize the resulting volatility smiles and 3D surfaces.

    -   **3_Market_Analytics:** An option chain explorer. Analyze volume, open interest,
        and implied volatility distributions for live market data.

    -   **4_Model_Fitting:** A page for analyzing historical return distributions.
        It includes a QQ-Plot to assess normality and a tool to fit jump-diffusion
        parameters from historical data using the method of moments.

    -   **5_Term_Structure:** A page for pricing and visualizing interest rate term
        structures. It allows users to price Zero-Coupon Bonds and generate yield
        curves using short-rate models like Vasicek and Cox-Ingersoll-Ross (CIR).

    This entire application is built on a robust library with a focus
    on speed, accuracy, and extensibility.
    """
)
