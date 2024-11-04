#!/bin/bash

# Création de l'environnement virtuel
echo "Création de l'environnement virtuel..."
python -m venv venv

# Activation de l'environnement virtuel
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

# Installation des dépendances
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Création des dossiers nécessaires
echo "Création des dossiers..."
mkdir -p cache
mkdir -p data
mkdir -p logs

echo "Installation terminée !"
echo "Pour lancer l'application, utilisez: ./run.sh"
