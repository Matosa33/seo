@echo off
setlocal enabledelayedexpansion

echo Verification de l'environnement...

:: Verification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python n'est pas installe. Installation requise.
    echo Telechargez Python depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Verification de Visual C++
where cl >nul 2>&1
if errorlevel 1 (
    echo Visual C++ Build Tools n'est pas installe.
    echo Installation requise: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo Installez "Desktop development with C++"
    pause
    
    :: On continue quand même avec une version allégée
    echo Installation en mode alternatif...
    goto install_lightweight
)

:install_full
echo Installation complete avec toutes les fonctionnalites...
goto install_common

:install_lightweight
echo Installation en mode allege...
echo (Certaines fonctionnalites seront limitees)

:install_common
:: Vérification de l'existence du fichier requirements.txt
if not exist requirements.txt (
    echo Le fichier requirements.txt est manquant!
    pause
    exit /b 1
)

:: Création et activation de l'environnement virtuel
echo Creation de l'environnement virtuel...
if exist venv (
    echo Suppression de l'ancien environnement virtuel...
    rmdir /s /q venv
)
python -m venv venv

:: Activation de l'environnement virtuel
echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

:: Installation des dépendances de base
echo Installation des dependances de base...
python -m pip install --upgrade pip
python -m pip install wheel setuptools

:: Installation depuis requirements.txt
echo Installation des packages depuis requirements.txt...
if errorlevel 1 (
    echo Installation des packages en mode allege...
    python -m pip install -r requirements.txt --only-binary :all:
) else (
    echo Installation complete des packages...
    python -m pip install -r requirements.txt
)

:: Création des dossiers nécessaires
echo Creation des dossiers...
mkdir cache 2>nul
mkdir data 2>nul
mkdir logs 2>nul

:: Vérification de l'installation
echo Verification de l'installation...
python -c "import streamlit; import pandas; import numpy; import plotly; import networkx; import gensim" >nul 2>&1
if errorlevel 1 (
    echo ATTENTION: Certains packages essentiels n'ont pas ete installes correctement.
    echo Verifiez les erreurs ci-dessus.
) else (
    echo Installation des packages reussie.
)

:: Création du fichier de lancement
echo Creation du fichier de lancement...
(
    echo @echo off
    echo call venv\Scripts\activate.bat
    echo streamlit run app.py
) > run.bat

echo Installation terminee !
echo Pour lancer l'application, utilisez: run.bat
pause