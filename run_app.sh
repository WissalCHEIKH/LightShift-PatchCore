#!/usr/bin/env bash
# Lance l'interface LightShift (Linux / macOS) : ./run_app.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Création de l'environnement Python .venv ..."
  python3 -m venv .venv
fi

echo "Installation / vérification des dépendances ..."
.venv/bin/python -m pip install -q -r requirements-app.txt

echo "Interface : http://localhost:8000   (Ctrl+C pour arrêter)"
exec .venv/bin/python -m uvicorn app.api.main:app --reload
