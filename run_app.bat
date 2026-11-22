@echo off
chcp 65001 >nul
REM Lance l'interface LightShift (Windows) : double-cliquer sur ce fichier.
REM 1re fois : crée l'environnement .venv et installe les dépendances (quelques minutes).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Création de l'environnement Python .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo Python est introuvable. Installez Python 3.11 ou plus depuis python.org, puis relancez.
        pause
        exit /b 1
    )
)

echo Installation / vérification des dépendances ...
".venv\Scripts\python.exe" -m pip install -q -r requirements-app.txt
if errorlevel 1 (
    echo L'installation des dépendances a échoué. Voir le message ci-dessus.
    pause
    exit /b 1
)

echo.
echo Interface : http://localhost:8000   (fermer cette fenêtre pour arrêter)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn app.api.main:app --reload
pause
