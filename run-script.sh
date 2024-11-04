#!/bin/bash

# Activation de l'environnement virtuel
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

# Lancement de l'application
echo "Démarrage de l'application..."
streamlit run app.py
