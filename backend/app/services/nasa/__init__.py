"""NASA Earth-observation API integrations (POWER, CMR, AppEEARS).

The legacy demonstration script ``02_fetch_power_api.py`` is kept unchanged; new
code should use :mod:`backend.app.services.nasa.power`.
"""

from .catalog import NASA_PRODUCTS, get_product
from .client import NasaHttpClient
from .errors import NasaApiError, NasaConfigurationError

__all__ = [
    "NASA_PRODUCTS",
    "NasaApiError",
    "NasaConfigurationError",
    "NasaHttpClient",
    "get_product",
]
