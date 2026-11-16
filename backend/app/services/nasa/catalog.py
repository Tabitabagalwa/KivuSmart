"""NASA products used by the Kivu restoration model.

Only NASA products are listed. HLSS30 (Sentinel-2 based) is intentionally absent
because the processing pipeline rejects Sentinel-derived satellite inputs.
Versions are the ones documented by the data centres at the time of writing;
override them in API calls if a newer version is published.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NasaProduct:
    key: str
    short_name: str
    version: str | None
    provider: str
    variables: tuple[str, ...]
    native_resolution: str
    role: str
    appeears_product: str | None = None
    appeears_layers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


NASA_PRODUCTS: dict[str, NasaProduct] = {
    product.key: product
    for product in (
        NasaProduct(
            key="nasadem",
            short_name="NASADEM_HGT",
            version="001",
            provider="LP DAAC",
            variables=("elevation",),
            native_resolution="30 m",
            role="Pente (%) et exposition ; pentes critiques > 35 %.",
            appeears_product="NASADEM_NC.001",
            appeears_layers=("NASADEM_HGT",),
        ),
        NasaProduct(
            key="srtm",
            short_name="SRTMGL1",
            version="003",
            provider="LP DAAC",
            variables=("elevation",),
            native_resolution="30 m",
            role="MNT alternatif pour pentes et bassins versants.",
            appeears_product="SRTMGL1_NC.003",
            appeears_layers=("SRTMGL1_DEM",),
        ),
        NasaProduct(
            key="hls_landsat",
            short_name="HLSL30",
            version="2.0",
            provider="LP DAAC",
            variables=("red", "nir", "ndvi"),
            native_resolution="30 m",
            role="Séries NDVI et calendrier des sols nus (Landsat harmonisé).",
        ),
        NasaProduct(
            key="modis_ndvi",
            short_name="MOD13Q1",
            version="061",
            provider="LP DAAC",
            variables=("ndvi", "evi"),
            native_resolution="250 m",
            role="Tendance NDVI 16 jours depuis 2000 (analyse de changement).",
            appeears_product="MOD13Q1.061",
            appeears_layers=("_250m_16_days_NDVI",),
        ),
        NasaProduct(
            key="modis_landcover",
            short_name="MCD12Q1",
            version="061",
            provider="LP DAAC",
            variables=("land_cover",),
            native_resolution="500 m",
            role="Occupation du sol annuelle (couche catégorielle).",
            appeears_product="MCD12Q1.061",
            appeears_layers=("LC_Type1",),
        ),
        NasaProduct(
            key="smap",
            short_name="SPL3SMP_E",
            version="006",
            provider="NSIDC",
            variables=("soil_moisture",),
            native_resolution="9 km",
            role="Déficit d'humidité du sol (priorité agroforesterie).",
            appeears_product="SPL3SMP_E.006",
            appeears_layers=("Soil_Moisture_Retrieval_Data_AM_soil_moisture",),
        ),
        NasaProduct(
            key="gpm_imerg",
            short_name="GPM_3IMERGM",
            version="07",
            provider="GES DISC",
            variables=("precipitation",),
            native_resolution="0.1° (~11 km)",
            role="Précipitations mensuelles et érosivité des pluies.",
        ),
        NasaProduct(
            key="gedi",
            short_name="GEDI02_A",
            version="002",
            provider="LP DAAC",
            variables=("canopy_height",),
            native_resolution="Empreintes de 25 m (échantillonnage discontinu)",
            role="Hauteur de canopée existante (potentiel RNA).",
        ),
        NasaProduct(
            key="ecostress",
            short_name="ECO_L2T_LSTE",
            version="002",
            provider="LP DAAC",
            variables=("land_surface_temperature",),
            native_resolution="70 m",
            role="Stress thermique des cultures non ombragées.",
        ),
        NasaProduct(
            key="nex_gddp_cmip6",
            short_name="NEX-GDDP-CMIP6",
            version="2.0",
            provider="NASA NCCS (Ames Research Center)",
            variables=("pr", "tas", "tasmax"),
            native_resolution="0.25° (~27 km), journalier 1950-2100",
            role="Scénarios climatiques CMIP6 (SSP1-2.6, SSP2-4.5, SSP5-8.5) : adaptation des essences "
                 "et du calendrier de plantation.",
        ),
        NasaProduct(
            key="landsat_c2_l2",
            short_name="Landsat Collection 2 Level-2",
            version="C2",
            provider="NASA / USGS",
            variables=("red", "nir", "qa_pixel"),
            native_resolution="30 m",
            role="NDVI de saison sèche 2015 et 2025 (Landsat 8-9 OLI).",
        ),
        NasaProduct(
            key="modis_burned_area",
            short_name="MCD64A1",
            version="061",
            provider="LP DAAC",
            variables=("burn_date",),
            native_resolution="500 m, mensuel",
            role="Fréquence des feux 2015-2025 (critère RNA).",
            appeears_product="MCD64A1.061",
            appeears_layers=("Burn_Date",),
        ),
    )
}


def get_product(key: str) -> NasaProduct:
    try:
        return NASA_PRODUCTS[key]
    except KeyError as exc:
        raise KeyError(
            f"Produit NASA inconnu : {key}. Disponibles : {', '.join(sorted(NASA_PRODUCTS))}"
        ) from exc
