from __future__ import annotations

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency for local dashboard runs
    st = None

from .engine import build_sample_parcels, recommend_restoration_strategy, summarize_strategy


def _prepare_parcels() -> list[dict]:
    return [
        {
            **parcel,
            "recommendation": recommend_restoration_strategy(parcel),
        }
        for parcel in build_sample_parcels()
    ]


def _render_dashboard() -> None:
    assert st is not None
    st.set_page_config(page_title="Kivu Smart Restoration", page_icon="🌱", layout="wide")
    st.title("Kivu Smart Restoration Dashboard")
    st.caption("Analyse décisionnelle de restauration des paysages montagneux du Sud-Kivu")

    parcels = _prepare_parcels()
    selected_name = st.selectbox("Parcelle pilote", [parcel["name"] for parcel in parcels])
    parcel = next(p for p in parcels if p["name"] == selected_name)
    recommendation = parcel["recommendation"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pente", f"{parcel['slope_pct']} %")
    col2.metric("NDVI", f"{parcel['ndvi']:.2f}")
    col3.metric("Distance forêt", f"{parcel['distance_to_forest_m']} m")

    st.subheader("Recommandation prioritaire")
    st.markdown(f"### {recommendation['strategy']}")
    st.write(recommendation["summary"])

    st.markdown("**Bénéfices attendus**")
    for benefit in recommendation["benefits"]:
        st.write(f"- {benefit}")

    st.markdown("**Risques et lacunes**")
    for risk in recommendation["risks"]:
        st.write(f"- {risk}")

    st.markdown("### Limites des données & validation locale")
    st.info(
        "La résolution spatiale des capteurs (de 10 m pour Sentinel-2/HLS à 9 km pour SMAP) nécessite une "
        "validation participative sur le terrain avec les agriculteurs et agronomes locaux avant toute plantation."
    )

    st.subheader("Vue synthèse des parcelles")
    st.dataframe(
        [
            {
                "Parcelle": p["name"],
                "Pente %": p["slope_pct"],
                "NDVI": round(p["ndvi"], 2),
                "Distance forêt m": p["distance_to_forest_m"],
                "Distance village km": p["distance_to_village_km"],
                "Décision": p["recommendation"]["strategy"],
            }
            for p in parcels
        ]
    )


def main() -> None:
    if st is None:
        print("Streamlit is not installed. To launch the dashboard, install dependencies from requirements.txt and run: streamlit run app.py")
        for parcel in _prepare_parcels():
            print(parcel["name"], summarize_strategy(parcel["recommendation"]))
        return
    _render_dashboard()


if __name__ == "__main__":
    main()
