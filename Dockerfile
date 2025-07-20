FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Instala dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Copia la carpeta de modelos si existe
COPY models/ models/

# Ejecuta el pipeline de entrenamiento para generar los modelos
RUN python src/training/pipeline.py

# Expone el puerto en el que correrá la API
EXPOSE 8080

# Comando para ejecutar el servidor de FastAPI con Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

