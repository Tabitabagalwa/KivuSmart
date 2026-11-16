from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.config import Settings
from ...db.models import Parcel, Recommendation
from ...db.repositories import get_parcel, latest_metrics
from ...decision_engine import ENGINE_VERSION, ahp_weights, evaluate_parcel, summarize_portfolio
from ...decision_engine.scoring_engine import STRATEGY_MODULES
from ..deps import get_db, get_rules, get_settings, require_api_key
from ..schemas import CODE_PATTERN, AHPRequest, DecisionRequest, PairwiseOverride

router = APIRouter(prefix="/decisions", tags=["aide à la décision"])
CodePath = Path(..., pattern=CODE_PATTERN)


def _reference(settings: Settings):
    return settings.data_dir / "reference" / "referentiel_kivu.json"


def _evaluate_stored(session: Session, parcel: Parcel, rules: dict, settings: Settings,
                     pairwise: dict | None = None) -> dict[str, Any]:
    stored = latest_metrics(session, parcel)
    evaluation = evaluate_parcel(stored["metrics"], rules=rules, pairwise=pairwise,
                                 reference_path=_reference(settings))
    demo = parcel.is_demo or any(p["is_demo"] for p in stored["provenance"].values())
    if demo:
        evaluation["notes"].insert(0, "Évaluation fondée sur des données de démonstration non validées.")
    return {"parcel": parcel.code, "name": parcel.name, "area_ha": parcel.area_ha,
            "based_on_demo_data": demo, "inputs": stored, "evaluation": evaluation}


@router.get("/rules", summary="Seuils et pondérations AHP par défaut")
def rules(rules: dict = Depends(get_rules)) -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "rules": rules,
        "criteria": {
            sid: {"order": list(m.CRITERIA_ORDER), "pairwise": m.DEFAULT_PAIRWISE,
                  "ahp": ahp_weights(m.DEFAULT_PAIRWISE).as_dict()}
            for sid, m in STRATEGY_MODULES.items()
        },
    }


@router.post("/ahp", summary="Calculer des poids AHP et le ratio de cohérence")
def ahp(body: AHPRequest) -> dict:
    return ahp_weights(body.matrix).as_dict()


@router.post("/evaluate", summary="Évaluer des valeurs saisies (simulation, rien n'est enregistré)")
def evaluate(body: DecisionRequest, rules: dict = Depends(get_rules),
             settings: Settings = Depends(get_settings)) -> dict:
    return evaluate_parcel(body.metrics.model_dump(exclude_none=True), rules=rules,
                           pairwise=body.pairwise, reference_path=_reference(settings))


@router.get("/summary", summary="Synthèse des recommandations sur toutes les parcelles")
def summary(
    limit: int = Query(1000, ge=1, le=5000),
    session: Session = Depends(get_db),
    rules: dict = Depends(get_rules),
    settings: Settings = Depends(get_settings),
) -> dict:
    parcels = list(session.scalars(select(Parcel).order_by(Parcel.code).limit(limit)))
    results = [_evaluate_stored(session, p, rules, settings) for p in parcels]
    return {
        **summarize_portfolio(results),
        "parcels": [
            {
                "parcel": r["parcel"],
                "name": r["name"],
                "area_ha": r["area_ha"],
                "decision_status": r["evaluation"]["decision_status"],
                "strategy_id": (r["evaluation"]["recommended"] or {}).get("strategy_id"),
                "score": (r["evaluation"]["recommended"] or {}).get("score"),
                "confidence": r["evaluation"]["confidence"],
                "based_on_demo_data": r["based_on_demo_data"],
            }
            for r in results
        ],
    }


@router.get("/parcels/{code}", summary="Évaluer une parcelle à partir de ses observations")
def parcel_decision(code: str = CodePath, session: Session = Depends(get_db),
                    rules: dict = Depends(get_rules), settings: Settings = Depends(get_settings)) -> dict:
    return _evaluate_stored(session, get_parcel(session, code), rules, settings)


@router.post("/parcels/{code}", dependencies=[Depends(require_api_key)],
             summary="Évaluer et enregistrer la recommandation (historique)")
def save_parcel_decision(
    body: PairwiseOverride | None = None,
    code: str = CodePath,
    session: Session = Depends(get_db),
    rules: dict = Depends(get_rules),
    settings: Settings = Depends(get_settings),
) -> dict:
    parcel = get_parcel(session, code)
    result = _evaluate_stored(session, parcel, rules, settings, body.pairwise if body else None)
    evaluation = result["evaluation"]
    recommended = evaluation["recommended"]
    record = Recommendation(
        parcel_id=parcel.id,
        strategy_id=recommended["strategy_id"] if recommended else evaluation["decision_status"],
        score=recommended["score"] if recommended else None,
        confidence=evaluation["confidence"],
        engine_version=ENGINE_VERSION,
        based_on_demo_data=result["based_on_demo_data"],
        result=result,
    )
    session.add(record)
    session.commit()
    return {"recommendation_id": record.id, **result}


@router.get("/parcels/{code}/history", summary="Historique des recommandations enregistrées")
def parcel_history(code: str = CodePath, session: Session = Depends(get_db)) -> list[dict]:
    parcel = get_parcel(session, code)
    rows = session.scalars(
        select(Recommendation).where(Recommendation.parcel_id == parcel.id).order_by(Recommendation.id.desc())
    )
    return [
        {"id": r.id, "strategy_id": r.strategy_id, "score": r.score, "confidence": r.confidence,
         "engine_version": r.engine_version, "based_on_demo_data": r.based_on_demo_data,
         "created_at": r.created_at}
        for r in rows
    ]
