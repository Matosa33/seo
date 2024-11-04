@echo off
echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

echo Demarrage de l'application...
streamlit run app\app.py
pause
