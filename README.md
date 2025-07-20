# ML Challenge - Titanic Classifier

## Descripción

Este proyecto implementa un sistema de clasificación binaria para predecir la supervivencia de pasajeros del Titanic, utilizando modelos entrenados sobre datos del conjunto `train.csv` (Titanic). Los modelos se generan a través de un pipeline de preprocesamiento y entrenamiento, y se exponen mediante una API RESTful en FastAPI para realizar inferencias en tiempo real. El sistema está contenerizado con Docker y automatizado mediante flujos CI/CD para garantizar su calidad, reproducibilidad y escalabilidad.

---

## Estructura del proyecto

```
.
├── data/                  # Dataset original (train.csv - no incluido)
├── models/                # Modelos entrenados (.pkl)
├── src/
│   ├── training/          # Preprocesamiento y entrenamiento
│   └── api/               # API en FastAPI
├── tests/                 # Pruebas unitarias con Pytest
├── .github/workflows/     # Workflows de CI/CD (GitHub Actions)
├── requirements.txt       # Dependencias del proyecto
└── Dockerfile             # Imagen Docker para la API
```

---

## Instalación y ejecución

### Requisitos

- Python 3.10+
- pip
- Docker

### Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Entrenamiento de modelos A y B

```bash
python src/training/pipeline.py
```

### Ejecución de la API

```bash
uvicorn src.api.main:app --reload
```

Documentación interactiva:
- http://localhost:8000/docs
- http://localhost:8000/redoc

---

## Ejemplo de inferencia usando el endpoint

### Con Python

```python
import requests

url = "http://localhost:8000/predict"
headers = {
    "x-api-key": "rappi-secret",
    "Content-Type": "application/json"
}
payload = [
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "C"
    }
]

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.json())
```

### Con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: rappi-secret" \
  -H "Content-Type: application/json" \
  -d '[{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}]'
```

### Obtener feature importances

```python
import requests

url = "http://localhost:8000/feature-importances"
headers = { "x-api-key": "rappi-secret" }

response = requests.get(url, headers=headers)
print(response.status_code)
print("Feature importances:", response.json())
```

---

## Evaluación del modelo

Se entrenaron y compararon dos versiones de modelo:

### Modelo A: Regresión Logística (inicial)

- Accuracy: 0.80  
- F1-score clase 0 (no sobreviviente): 0.84  
- F1-score clase 1 (sobreviviente): 0.74  
- Macro F1-score: 0.79

Este modelo sirve como referencia inicial, con desempeño razonable pero sesgo hacia la clase mayoritaria.

### Modelo B: Gradient Boosting (mejorado)

- Accuracy: 0.83  
- F1-score clase 0: 0.86  
- F1-score clase 1: 0.78  
- Macro F1-score: 0.82

El modelo B mejora recall, precisión y F1-score en ambas clases, siendo más balanceado y robusto para producción.

Además:

- Tiempo total de entrenamiento: 1.36 s  
- Memoria máxima: 113.66 MB  
- Uso CPU: 36.6%

### Análisis de profiling

Durante la ejecución del pipeline se recolectaron métricas clave de rendimiento:

- **Tiempo total de ejecución:** 1.36 segundos  
- **Uso pico de memoria:** 113.66 MB  
- **Uso medio de CPU:** 36.6%

Estas métricas reflejan un pipeline altamente eficiente y apto para entornos productivos con recursos limitados. Algunos puntos a destacar:

- El **tiempo de ejecución** es inferior a 2 segundos, lo que permite integrarlo fácilmente en flujos CI/CD y en endpoints que requieran reentrenamiento frecuente o automatizado.
- El **uso de memoria** está contenido (< 120 MB), por lo que el pipeline puede ejecutarse sin problemas en instancias pequeñas, contenedores serverless o incluso dispositivos edge.
- El uso de **CPU moderado** indica que las operaciones más pesadas (transformaciones, entrenamiento de modelos y serialización) están bien optimizadas y distribuidas.

El profiling está integrado automáticamente en el pipeline mediante `psutil` y `memory_profiler`, por lo que cualquier ajuste futuro o incorporación de nuevos modelos podrá ser monitoreado de manera continua sin overhead adicional.

---

## Features más importantes

Importancias (log-odds) extraídas del modelo:

- **Sex_female**: +1.31 - ser mujer incrementa la probabilidad de supervivencia.  
- **Sex_male**: -1.30 - ser hombre reduce fuertemente la probabilidad de supervivencia.  
- **Pclass_3**: -0.98 - la tercera clase representa las probabilidades más bajas de sobrevivir.  
- **Pclass_1**: +0.70 - primera clase aumenta las probabilidades de supervivencia.  
- **Pclass_2**: +0.28 - es mayor la probabilidad de sobrevivir que de fallecer.  
- **Embarked_S**: -0.27 - embarcar en Southampton representa reduce las probabilidades de supervivencia.  
- **SibSp**: -0.27 - tener hermanos/pareja a bordo impacta negativamente las probabilidades de sobrevivir.  
- **Embarked_C**: +0.23 - embarcar en Cherbourg mejora la probabilidad de supervivencia.  
- **Parch**: -0.14 - tener padres/hijos tiene leve impacto negativo.  
- **Embarked_Q**: +0.05 - no representa una relación significativa con la supervivencia.  
- **Age**: -0.03 - no representa una relación significativa con la supervivencia.
- **Fare**: +0.005 - no representa una relación significativa con la supervivencia.

Las variables más representativas en las clasificaciones combinan factores socioeconómicos y demográficos, este comportamiento es coherente con los hechos históricos del naufragio del Titanic.

---

## Pruebas automáticas

```bash
pytest tests/
```

- Validan predicciones, respuesta de endpoints y errores esperados.  
- Diseñadas para entornos CI/CD con alta cobertura.

---

## CI/CD y despliegue

### CI: GitHub Actions

- Corre con cada push/PR a `main`  
- Ejecuta tests + entrenamiento + build Docker

### CD (opcional)

- Soporte para despliegue en Google Cloud Run  
- Preparado para integrarse con secretos en GitHub

### Docker

```bash
docker build -t titanic-api .
docker run -p 8080:8080 titanic-api
```

---

## Funcionalidades de la aplicación

- Dos modelos entrenados y comparados (Logistic y Boosting)  
- Pipeline modular y automatizado  
- API REST con FastAPI  
- Validación robusta con Pydantic  
- Seguridad con API-Key  
- Logs y profiling integrados  
- Endpoint de importancias (`/feature-importances`)  
- A/B Testing soportado  
- Testing automático (Pytest)  
- CI/CD con GitHub Actions  
- Contenerización con Docker

---

## Conclusión

### Métricas de evaluación

El modelo B logró una mejora observable respecto al baseline, alcanzando un F1-score de 0.82 frente al 0.80 del modelo A. Esta diferencia, aunque no es considerable, sí es consistente y evidencia una mejora en la capacidad predictiva, sobre todo en situaciones desbalanceadas.

### Insights del modelo

- Las variables socioeconómicas (clase, tarifa pagada) y demográficas (género, edad) tienen la influencia más representativa en la predicción generada por el modelo.  
- El comportamiento observado es representa de forma coherente el evento real.  
- Se puede concluir que, incluso con un dataset limitado, modelos bien planteados, entrenados y analizados pueden ofrecer valor para resolver tareas complejas.

### Producción y MLOps

- El entrenamiento es independiente del despliegue para permitir mayor flexibilidad en el proceso.  
- Los modelos se serializan con `joblib` y se cargan de forma dinámica al iniciar la API.  
- El proyecto es portable, reproducible y escalable (por ejemplo con kubernetes o cloud run).  
- La arquitectura del proyecto facilita la integración de nuevos modelos, incluso nuevos endpoint para la API, así mismo facilita la puesta en producción del mismo.

---

## Autor

**Cristian Arbelaez**  
GitHub: [https://github.com/cdarbelaez](https://github.com/cdarbelaez)
