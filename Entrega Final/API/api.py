# -*- coding: utf-8 -*-
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Variables globales
modelo = None
scaler = None
COLUMNAS_RELEVANTES = ["LIMIT_BAL", "AGE", "PAY_0", "SEX", "EDUCATION", "MARRIAGE"]

# Entrenar el modelo y el escalador
datos_ficticios = pd.DataFrame({
    "LIMIT_BAL": [200000, 50000, 300000],
    "AGE": [30, 50, 40],
    "PAY_0": [0, 2, -1],
    "SEX": [1, 2, 1],
    "EDUCATION": [2, 1, 3],
    "MARRIAGE": [1, 2, 2],
    "default.payment.next.month": [0, 1, 0]
})

scaler = StandardScaler()
X = datos_ficticios[COLUMNAS_RELEVANTES]
y = datos_ficticios["default.payment.next.month"]
X_scaled = scaler.fit_transform(X)

modelo = LogisticRegression(max_iter=500, penalty="l2", solver="saga")
modelo.fit(X_scaled, y)

# Clase para datos de entrada
class InputData(BaseModel):
    LIMIT_BAL: float
    AGE: float
    PAY_0: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int

# Crear un router para incluirlo en main.py
router = APIRouter()

@router.post("/predict")
def predict(data: InputData):
    try:
        nueva_data = pd.DataFrame([[data.LIMIT_BAL, data.AGE, data.PAY_0,
                                    data.SEX, data.EDUCATION, data.MARRIAGE]],
                                  columns=COLUMNAS_RELEVANTES)
        nueva_data_scaled = scaler.transform(nueva_data)
        probabilidad = modelo.predict_proba(nueva_data_scaled)[0][1]
        riesgo = (
            "BAJO" if probabilidad <= 0.35
            else "MEDIO" if probabilidad <= 0.65
            else "ALTO"
        )
        return {"probabilidad": round(probabilidad, 2), "riesgo": riesgo}
    except Exception as e:
        return {"error": f"No se pudo procesar la predicción: {e}"}



