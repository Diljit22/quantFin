from demo._utils._compare_tech import compare_techniques
from src.models.black_scholes_merton import BlackScholesMerton
from src.techniques.characteristic.fft_technique import FourierPricingTechnique
from src.techniques.closed_forms.bsm_technique import BlackScholesMertonTechnique
from src.techniques.closed_forms.bsm_finite_diff import FD_BSM
from src.techniques.characteristic.integration_technique import IntegrationTechnique
from src.techniques.pde.pde_techique import PDETechnique


def view_bsm():
    # 1) Create a single model
    model = BlackScholesMerton(sigma=0.20)

    # 2) List multiple techniques
    techniques_list = [
        BlackScholesMertonTechnique(cache_results=False),
        FD_BSM(cache_results=False),
        IntegrationTechnique(cache_results=False),
        PDETechnique(S_max=600.0, M=256, N=256),
        FourierPricingTechnique(),
    ]

    # 3) Run the comparison
    compare_techniques(techniques_list, model)
