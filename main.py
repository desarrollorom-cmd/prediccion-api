from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API Predicción Ventas WIN")

modelo = joblib.load("modelo_ventas_win.pkl")

class PrediccionRequest(BaseModel):
    pdv_trafico: int
    dias_laborados: int
    experiencia_meses: int
    visitas_supervisor: int
    promociones_activas: int

class PrediccionResponse(BaseModel):
    ventas_estimadas: float

@app.post("/prediccion/ventas", response_model=PrediccionResponse)
def predecir_ventas(data: PrediccionRequest):
    X = np.array([[
        data.pdv_trafico,
        data.dias_laborados,
        data.experiencia_meses,
        data.visitas_supervisor,
        data.promociones_activas
    ]])

    pred = modelo.predict(X)[0]
    return {"ventas_estimadas": round(pred, 2)}
