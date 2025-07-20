from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import joblib
import pandas as pd
import logging
import time
import os
import random

# Configuración de seguridad
API_KEY = "rappi-secret"  # Cambiar por una clave segura en producción

# Configuración de logs
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

# Inicialización de la aplicación
app = FastAPI(title="Titanic Survival Predictor API", version="1.0")


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware para medir el tiempo de respuesta de cada petición HTTP."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logging.info(f"{request.method} {request.url.path} - {duration:.3f}s")
        return response


app.add_middleware(TimingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Maneja errores de validación del modelo de entrada.

    Args:
        request (Request): Objeto de solicitud HTTP.
        exc (RequestValidationError): Excepción de validación.

    Returns:
        JSONResponse: Respuesta JSON con detalles del error.
    """
    logging.error(f"Error de validación: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# ============================
# Carga de modelos A y B
# ============================

model_A = joblib.load("models/model_A.pkl")
model_B = joblib.load("models/model_B.pkl")

def choose_model():
    """Selecciona aleatoriamente entre el modelo A y el modelo B.

    Returns:
        dict: Diccionario con 'model' y 'features'.
    """
    seleccion = random.choice(["A", "B"])
    seleccionado = model_A if seleccion == "A" else model_B
    logging.info(f"Modelo seleccionado: {seleccion}")
    return seleccionado

# ============================
# Modelo de entrada
# ============================

class Passenger(BaseModel):
    """Modelo de entrada que representa un pasajero del Titanic.

    Attributes:
        Pclass (int): Clase del pasajero (1, 2 o 3).
        Sex (str): Género ('male' o 'female').
        Age (float): Edad en años.
        SibSp (int): Nº de hermanos/esposas a bordo.
        Parch (int): Nº de padres/hijos a bordo.
        Fare (float): Tarifa pagada.
        Embarked (str): Puerto de embarque ('C', 'Q', 'S').
    """
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# ============================
# Endpoint de predicción
# ============================

@app.post("/predict")
def predict(passengers: list[Passenger], x_api_key: str = Header(...)):
    """Predice la supervivencia de una lista de pasajeros usando A/B Testing.

    Args:
        passengers (list[Passenger]): Lista de objetos Passenger.
        x_api_key (str): API Key enviada en el header de la solicitud.

    Returns:
        dict: Diccionario con predicciones binarias (0 = no sobrevivió, 1 = sí sobrevivió).

    Raises:
        HTTPException: 401 si la API Key es inválida, 500 si hay un error interno.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

    try:
        logging.info(f"Petición recibida para {len(passengers)} pasajeros")

        # Selección del modelo (A o B)
        model_data = choose_model()
        model = model_data["model"]
        features = model_data["features"]

        df = pd.DataFrame([p.dict() for p in passengers])
        df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=False)

        for col in features:
            if col not in df.columns:
                df[col] = 0

        df = df[features]
        preds = model.predict(df)

        logging.info("Predicción completada correctamente")
        return {"predictions": preds.tolist()}

    except Exception as e:
        logging.exception("Error durante la predicción")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint de feature importances
# ============================

@app.get("/feature-importances")
def feature_importances(x_api_key: str = Header(...)):
    """Devuelve las importancias de cada feature del modelo A.

    Args:
        x_api_key (str): API Key enviada en el header de la solicitud.

    Returns:
        dict: Diccionario ordenado por magnitud con las importancias.

    Raises:
        HTTPException: 401 si la API Key es inválida, 400 si el modelo no tiene coeficientes.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

    try:
        importances = model_A["model"].coef_[0]
        result = dict(zip(model_A["features"], importances))
        sorted_result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
        logging.info("Importancias de features calculadas correctamente (modelo A)")
        return sorted_result

    except AttributeError:
        logging.error("El modelo A no soporta coeficientes (coef_)")
        raise HTTPException(status_code=400, detail="Este modelo no soporta importancias por coeficientes.")

    except Exception as e:
        logging.exception("Error al obtener importancias")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint de salud
# ============================

@app.get("/health")
def health():
    """Verifica que la API esté operativa.

    Returns:
        dict: {"status": "ok"} si la API funciona correctamente.
    """
    return {"status": "ok"}