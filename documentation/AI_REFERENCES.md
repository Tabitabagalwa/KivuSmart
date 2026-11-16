# AI References

This file logs the AI-assisted workflow used for the Kivu restoration prototype.

- 2026-11-15: Reviewed the challenge brief and extracted the key land-restoration decision rules from the NASA Space Apps documentation.
- 2026-11-15: Designed the three main restoration pathways: RNA, reboisement pur, and agroforesterie intelligente.
- 2026-11-15: Implemented a lightweight Python scoring engine and a dashboard prototype based on the scientific constraints described in the brief.

- 2026-11-15: Used Claude Code (Anthropic) to audit the backend and extend it: FastAPI API, PostgreSQL/PostGIS persistence, NASA POWER/CMR/AppEEARS clients, data import, geoprocessing services, change analysis (raster difference, Mann-Kendall/Sen), an AHP-weighted fuzzy decision engine and an offline test suite. Fixed an affine-version incompatibility (`Affine @ tuple`) and a swapped row/column index in raster resolution sampling. Decision thresholds come from the challenge brief and `referentiel_kivu.json`; they remain hypotheses to validate in the field.

The project intentionally keeps the decision logic transparent and references the scientific caveat that field validation remains mandatory before deployment.
