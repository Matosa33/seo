@echo off
setlocal enabledelayedexpansion

echo =================================
echo Installation du Cocon Semantique
echo =================================

:: Verification de l'environnement
echo [1/7] Verification de l'environnement...

:: Verification de Python avec version minimale
python --version > temp.txt 2>&1
set /p PYTHON_VERSION=<temp.txt
del temp.txt
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe.
    echo Telechargez Python 3.9+ depuis https://www.python.org/downloads/
    echo Assurez-vous de cocher "Add Python to PATH" lors de l'installation
    pause
    exit /b 1
) else (
    echo [OK] Python detecte: %PYTHON_VERSION%
)

:: Verification de Git
git --version > nul 2>&1
if errorlevel 1 (
    echo [INFO] Git n'est pas installe. L'installation continuera sans Git.
) else (
    echo [OK] Git detecte
)

:: Verification de Visual C++
where cl >nul 2>&1
if errorlevel 1 (
    echo [INFO] Visual C++ Build Tools non detecte
    echo Installation des packages en mode compatible...
    set LIGHTWEIGHT_INSTALL=1
) else (
    echo [OK] Visual C++ Build Tools detecte
    set LIGHTWEIGHT_INSTALL=0
)

:: Creation de l'environnement virtuel
echo [2/7] Creation de l'environnement virtuel...
if exist venv (
    echo [INFO] Suppression de l'ancien environnement virtuel...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo [ERREUR] Impossible de creer l'environnement virtuel
    pause
    exit /b 1
)

:: Activation de l'environnement
echo [3/7] Activation de l'environnement...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)

:: Mise a jour de pip et outils de base
echo [4/7] Installation des outils de base...
python -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo [ATTENTION] Erreur lors de la mise a jour des outils de base
)

:: Installation des dependances
echo [5/7] Installation des packages...
if %LIGHTWEIGHT_INSTALL%==1 (
    echo [INFO] Installation en mode compatible...
    python -m pip install --only-binary :all: -r requirements.txt
) else (
    python -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo [ATTENTION] Certains packages n'ont pas pu etre installes
)

:: Creation des dossiers
echo [6/7] Creation des dossiers necessaires...
mkdir cache 2>nul
mkdir data 2>nul
mkdir logs 2>nul

:: Verification de l'installation
echo [7/7] Verification de l'installation...
python -c "import streamlit; import pandas; import numpy; import plotly; import networkx; import gensim" >nul 2>&1
if errorlevel 1 (
    echo [ATTENTION] Certains packages essentiels n'ont pas ete installes correctement.
    echo Verifiez les erreurs ci-dessus.
) else (
    echo [OK] Packages principaux installes avec succes
)

:: Creation du fichier de lancement
echo @echo off > run.bat
echo title Cocon Semantique >> run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo streamlit run app.py >> run.bat

echo =================================
echo Installation terminee !
echo =================================
echo.
echo Pour lancer l'application:
echo 1. Double-cliquez sur run.bat
echo - ou -
echo 2. Ouvrez une console et tapez: run.bat
echo.
echo En cas de probleme:
echo - Verifiez que Python 3.9+ est installe
echo - Installez Visual C++ Build Tools si necessaire
echo - Consultez le fichier logs/install.log
echo.
pause