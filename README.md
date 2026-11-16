# Kivu Smart

> **Démonstration : [regarder la vidéo de démonstration](https://youtube.com/shorts/_Bt3p_2nl1I?si=H7oA7QCh-xxzygOL)**

Kivu Smart est une plateforme d'aide à la décision pour la restauration des paysages
du territoire de Kalehe, au Sud-Kivu (République démocratique du Congo). Elle combine
cartographie interactive, traitement géospatial, données d'observation de la Terre et
analyse multicritère pour aider à identifier les zones prioritaires et à comparer
des stratégies de restauration.

> **Important :** le jeu de données livré dans ce dépôt est un jeu de démonstration
> synthétique. Il sert à tester l'interface et la chaîne de traitement ; il ne
> constitue ni une mesure NASA ni un relevé de terrain. Toute recommandation doit
> être vérifiée localement avec les agronomes, les communautés et les autorités
> compétentes avant toute intervention.

## Fonctionnalités

- tableau de bord de la zone d'étude et indicateurs de restauration ;
- carte interactive des parcelles, couches raster et zones d'intervention ;
- analyse des changements de végétation et des tendances temporelles ;
- comparaison de trois stratégies : régénération naturelle assistée (RNA),
  agroforesterie et reboisement ;
- simulateur de décision fondé sur les règles métier et la méthode AHP ;
- consultation de scénarios climatiques futurs CMIP6 ;
- chaîne de traitement géospatial : reprojection, grille commune, pente,
  exposition, NDVI, statistiques zonales et proximités ;
- API FastAPI pour l'import de données, les analyses, les décisions et les services
  NASA (POWER, CMR, AppEEARS et CMIP6) ;
- stockage PostgreSQL/PostGIS pour les usages persistants et Docker Compose pour
  lancer la base de données.

## Architecture

```text
KivuSmart/
├── app.py                         # Point d'entrée Streamlit
├── frontend/                      # Interface et pages du tableau de bord
├── backend/app/
│   ├── server.py                  # Application FastAPI
│   ├── api/routers/               # Routes REST
│   ├── decision_engine/           # Règles, AHP et scores de restauration
│   ├── processing/                # Traitements raster et vecteur
│   ├── services/nasa/              # Clients et catalogues NASA
│   └── db/                        # Modèles, sessions et dépôts PostGIS
├── config/                        # Zone d'étude, couches et règles
├── data/                          # Données simulées, réelles et produites
├── scripts/                       # Préparation et téléchargement des données
└── documentation/                # Références scientifiques et IA
```

Le front-end Streamlit peut fonctionner hors ligne avec les données présentes dans
`data/simulated/`. Le backend FastAPI est indépendant et utilise PostgreSQL/PostGIS
pour les traitements et les données persistantes.

## Prérequis

- Python 3.10 ou plus récent ;
- `pip` et, pour la base complète, Docker Desktop avec Docker Compose ;
- Git (facultatif, pour cloner le dépôt) ;
- au moins 4 Go de mémoire recommandés pour les traitements géospatiaux.

Les dépendances Python sont listées dans
[`requirements.txt`](requirements.txt). Elles couvrent notamment Streamlit,
GeoPandas, Rasterio, Folium, FastAPI, SQLAlchemy, GeoAlchemy2, psycopg et
`earthengine-api`.

## Installation

Depuis le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si l'exécution est bloquée par la politique PowerShell, utiliser l'invite de
commandes avec `.venv\Scripts\activate.bat`, ou lancer PowerShell avec la politique
adaptée à l'environnement de développement.

## Lancer l'interface Streamlit

Depuis la racine du dépôt :

```powershell
streamlit run KivuSmart\app.py
```

Ou depuis le dossier `KivuSmart` :

```powershell
cd KivuSmart
streamlit run app.py
```

L'interface est ensuite disponible par défaut à
`http://localhost:8501`. Les pages principales sont :

1. Tableau de bord ;
2. Carte des zones ;
3. Analyse des zones ;
4. Comparaison des stratégies ;
5. Simulateur de décision ;
6. Climat futur (CMIP6) ;
7. Méthodologie et données.

## Lancer l'API FastAPI et PostGIS

Depuis `KivuSmart`, démarrer PostgreSQL/PostGIS :

```powershell
docker compose up -d db
```

Initialiser la base avec les parcelles de démonstration :

```powershell
python -m backend.app.db.init_db --demo
```

Lancer l'API :

```powershell
uvicorn backend.app.server:app --reload
```

L'API est disponible à `http://127.0.0.1:8000`. La documentation interactive est
accessible à [`/docs`](http://127.0.0.1:8000/docs) et la documentation ReDoc à
[`/redoc`](http://127.0.0.1:8000/redoc).

Principaux groupes de routes sous `/api/v1` :

- `/health` et `/health/ready` : état du service et de la base ;
- `/parcels` : parcelles, GeoJSON, métriques et observations ;
- `/imports` : import GeoJSON/GPKG/Shapefile, CSV et données de démonstration ;
- `/datasets` : jeux de données enregistrés ;
- `/processing` : pipeline, NDVI, terrain, proximités et statistiques zonales ;
- `/analysis` : changements, tendances et anomalies climatiques ;
- `/decisions` : règles, AHP, scores, synthèses et historiques ;
- `/nasa` : produits NASA, NASA POWER, CMIP6, CMR et AppEEARS.

Les routes d'écriture peuvent exiger l'en-tête `X-API-Key`.

## Configuration

Copier [`.env.example`](KivuSmart/.env.example) en `.env` dans `KivuSmart/`, puis
adapter les valeurs nécessaires. Les paramètres backend utilisent le préfixe
`KIVU_`, notamment :

| Variable | Rôle | Valeur de développement |
| --- | --- | --- |
| `KIVU_DATABASE_URL` | URL PostgreSQL/PostGIS | `postgresql+psycopg://kivu:kivu@localhost:5432/kivu_smart` |
| `KIVU_API_KEY` | Clé des opérations protégées | non définie en développement |
| `KIVU_ENVIRONMENT` | Environnement d'exécution | `development` |
| `KIVU_CORS_ORIGINS` | Origines autorisées | localhost:8501 |
| `KIVU_ALLOWED_HOSTS` | Hôtes autorisés | `*` en développement |
| `KIVU_DOCS_ENABLED` | Active `/docs` et `/redoc` | `true` |
| `KIVU_EARTHDATA_TOKEN` | Jeton NASA Earthdata | optionnel |
| `KIVU_EARTHDATA_USERNAME` / `KIVU_EARTHDATA_PASSWORD` | Identifiants NASA | optionnels |

En production, une clé `KIVU_API_KEY` d'au moins 24 caractères, PostgreSQL/PostGIS
et des origines CORS explicites sont requis. Ne jamais versionner un fichier `.env`
contenant des secrets.

## Données et méthode

La chaîne de décision suit les étapes suivantes :

1. définition de la zone de Kalehe et harmonisation des projections ;
2. préparation des facteurs (altitude, pente, NDVI, couvert arboré, feux,
   humidité, pluie et occupation du sol) ;
3. calcul des distances aux villages, routes, cours d'eau, champs et aires
   protégées ;
4. notation des critères sur une échelle de 1 à 5 ;
5. application de filtres et pondérations AHP ;
6. production de cartes d'aptitude, zones prioritaires et scénarios.

Les sources prévues ou documentées comprennent NASADEM, Landsat 8-9, Global Forest
Change, MODIS MCD64A1, NASA POWER, NEX-GDDP-CMIP6, OCHA/geoBoundaries et
OpenStreetMap. Les références détaillées sont disponibles dans
[`documentation/scientific_references.md`](KivuSmart/documentation/scientific_references.md)
et [`documentation/AI_REFERENCES.md`](KivuSmart/documentation/AI_REFERENCES.md).

Pour préparer des données réelles ou télécharger des produits, consulter :

```powershell
python scripts\preparer_donnees_reelles.py --help
python scripts\telecharger_donnees_nasa.py --help
python scripts\telecharger_cmip6.py --help
```

Le jeu synthétique peut être régénéré avec :

```powershell
python scripts\generer_donnees_simulees.py
```

## Tests

Depuis le dossier `KivuSmart` :

```powershell
python -m unittest discover -s backend\tests -p "test_*.py"
python -m unittest discover -s frontend\tests -p "test_*.py"
```

Les tests PostGIS nécessitent une base démarrée par Docker et une configuration
adaptée. Les tests NASA utilisent des services simulés et ne doivent pas nécessiter
de secrets pour la suite hors ligne.

## Dépannage rapide

- **La carte est vide** : vérifier que les fichiers de
  `data\simulated\rasters` et `data\simulated\vecteurs` sont présents et régénérer
  les données synthétiques.
- **L'API ne démarre pas** : vérifier l'environnement virtuel, le port 8000 et
  `KIVU_DATABASE_URL`.
- **PostGIS n'est pas prêt** : exécuter `docker compose ps`, attendre le
  healthcheck, puis relancer l'initialisation.
- **Les données NASA ne sont pas disponibles** : vérifier les identifiants
  Earthdata, la connexion réseau et les limites des API.

## Utilisation responsable et licence des données

Kivu Smart est un outil d'aide à la décision, et non un système de validation
automatique de projets de restauration. Les résolutions, dates, incertitudes et
lacunes d'OpenStreetMap doivent être prises en compte. Les données externes restent
soumises aux licences et conditions de leurs fournisseurs, notamment l'ODbL pour
OpenStreetMap et les conditions d'accès NASA.