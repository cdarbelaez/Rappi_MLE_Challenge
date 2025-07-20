import pandas as pd
import time
import psutil
import os
import logging
from data_preparation import preprocess_data
from model import train_and_export_model

# Configurar logs
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)

# Crear carpeta para almacenar los modelos
os.makedirs("models", exist_ok=True)

def run_pipeline():
    """
    Ejecuta el pipeline completo de entrenamiento para dos modelos (A y B).

    Este pipeline incluye:
    - Carga del dataset Titanic desde un archivo CSV.
    - Preprocesamiento del dataset con limpieza e ingeniería de features.
    - Entrenamiento de dos versiones del modelo (A y B) con diferentes configuraciones.
    - Serialización de ambos modelos junto con las columnas utilizadas.
    """
    logging.info("Iniciando pipeline de entrenamiento")

    # Cargar el dataset original
    logging.info("Cargando dataset desde 'data/train.csv'")
    df = pd.read_csv("data/train.csv")

    # Aplicar preprocesamiento
    logging.info("Aplicando preprocesamiento al dataset")
    df = preprocess_data(df)

    # Entrenar y guardar modelo A (baseline)
    logging.info("Entrenando y exportando modelo A (baseline)")
    train_and_export_model(df, version="A")

    # Entrenar y guardar modelo B (mejorado)
    logging.info("Entrenando y exportando modelo B (mejorado)")
    train_and_export_model(df, version="B")

    logging.info("Pipeline completado exitosamente")

def profile_execution(start_time):
    """
    Mide uso de CPU, memoria y tiempo al final del pipeline.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    cpu_percent = psutil.cpu_percent(interval=1)
    duration = time.time() - start_time

    logging.info(f"Tiempo total de ejecución: {duration:.2f} segundos")
    logging.info(f"Uso de memoria: {mem_mb:.2f} MB")
    logging.info(f"Uso de CPU: {cpu_percent:.2f} %")

if __name__ == "__main__":
    start = time.time()
    run_pipeline()
    profile_execution(start)
