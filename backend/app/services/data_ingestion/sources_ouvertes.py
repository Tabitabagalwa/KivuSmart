"""Données ouvertes non NASA nécessaires au contexte local (limites, OpenStreetMap).

* Limite administrative : geoBoundaries (gbOpen, ADM2 = territoires de la RDC), qui
  redistribue le Référentiel géographique commun d'OCHA RDC ;
* Contexte : OpenStreetMap via l'API Overpass (localités, routes, cours d'eau, champs
  cartographiés et limite du Parc national de Kahuzi-Biega).

Seuls les hôtes listés dans ``HOTES_AUTORISES`` sont contactés, en HTTPS.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

HOTES_AUTORISES = {
    "www.geoboundaries.org", "github.com", "raw.githubusercontent.com", "media.githubusercontent.com",
    "overpass-api.de", "overpass.private.coffee", "maps.mail.ru",
}
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
USER_AGENT = "KivuSmartRestoration/2.0 (NASA Space Apps research prototype)"


class SourceOuverteErreur(RuntimeError):
    pass


def _verifier(url: str) -> None:
    morceaux = urlsplit(url)
    if morceaux.scheme != "https" or (morceaux.hostname or "") not in HOTES_AUTORISES:
        raise SourceOuverteErreur(f"Hôte non autorisé : {morceaux.hostname}")


def client(timeout_s: float = 400.0) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0), follow_redirects=True,
                        headers={"User-Agent": USER_AGENT})


def _get(http: httpx.Client, url: str, **kwargs) -> httpx.Response:
    _verifier(url)
    reponse = http.get(url, **kwargs)
    _verifier(str(reponse.url))
    reponse.raise_for_status()
    return reponse


def limite_geoboundaries(http: httpx.Client, pays: str = "COD", niveau: str = "ADM2",
                         nom: str = "Kalehe") -> dict[str, Any]:
    """Entité GeoJSON de l'unité administrative ``nom`` (comparaison insensible à la casse)."""
    meta = _get(http, f"https://www.geoboundaries.org/api/current/gbOpen/{pays}/{niveau}/").json()
    collection = _get(http, meta["gjDownloadURL"]).json()
    trouvees = [f for f in collection.get("features", [])
                if str(f.get("properties", {}).get("shapeName", "")).casefold() == nom.casefold()]
    if not trouvees:
        raise SourceOuverteErreur(f"Unité « {nom} » absente de geoBoundaries {pays} {niveau}.")
    entite = trouvees[0]
    entite["properties"] = {
        **entite.get("properties", {}),
        "source": f"geoBoundaries gbOpen {pays} {niveau} — {meta.get('boundarySource', 'source non précisée')}",
        "annee_reference": meta.get("boundaryYearRepresented"),
        "licence": meta.get("boundaryLicense"),
    }
    return entite


def overpass(http: httpx.Client, requete: str, essais: int = 2) -> dict[str, Any]:
    """Exécute une requête Overpass QL (JSON) en essayant plusieurs serveurs publics."""
    derniere = "aucun serveur"
    for _ in range(essais):
        for url in OVERPASS:
            _verifier(url)
            try:
                reponse = http.post(url, data={"data": requete})
            except httpx.HTTPError as exc:
                derniere = f"{url} : {type(exc).__name__}"
                continue
            if reponse.status_code == 200:
                try:
                    return reponse.json()
                except json.JSONDecodeError:
                    derniere = f"{url} : réponse non JSON"
                    continue
            derniere = f"{url} : HTTP {reponse.status_code}"
        time.sleep(20)
    raise SourceOuverteErreur(f"Overpass indisponible ({derniere}).")


def emprise_overpass(ouest: float, sud: float, est: float, nord: float) -> str:
    return f"{sud:.5f},{ouest:.5f},{nord:.5f},{est:.5f}"
