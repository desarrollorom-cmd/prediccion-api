import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Datos simulados (luego puedes reemplazar por SQL Server)
data = {
    "pdv_trafico": [100, 200, 150, 300, 250],
    "dias_laborados": [20, 22, 18, 25, 23],
    "experiencia_meses": [3, 6, 12, 2, 8],
    "visitas_supervisor": [2, 4, 3, 5, 4],
    "promociones_activas": [1, 1, 0, 1, 0],
    "ventas_realizadas": [30, 55, 40, 80, 60]
}

df = pd.DataFrame(data)

X = df.drop("ventas_realizadas", axis=1)
y = df["ventas_realizadas"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
print("R2:", r2_score(y_test, predicciones))

# Guardar modelo
joblib.dump(modelo, "modelo_ventas_win.pkl")
