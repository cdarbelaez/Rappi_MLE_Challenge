import pandas as pd
import joblib
from src.training.model import train_and_export_model
from src.training.data_preparation import preprocess_data


def test_train_and_export_model(tmp_path):
    """
    Verifica que los modelos (versión A y B) se entrenen correctamente y se guarden.

    Utiliza un dataset simulado, lo preprocesa, entrena ambos modelos y
    valida que los archivos serializados se creen exitosamente.
    """
    # Dataset simulado
    data = {
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 2, 3, 2],
        "Sex": ["male", "female", "female", "male", "female"],
        "Age": [22, 38, 26, 35, 27],
        "SibSp": [1, 1, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Fare": [7.25, 71.2833, 7.925, 8.05, 13.00],
        "Embarked": ["S", "C", "S", "S", "Q"],
        "Name": ["A", "B", "C", "D", "E"],
        "Ticket": ["X", "Y", "Z", "K", "L"],
        "Cabin": [None, None, None, None, None],
        "PassengerId": [1, 2, 3, 4, 5]
    }

    df = pd.DataFrame(data)
    df_clean = preprocess_data(df)

    for version in ["A", "B"]:
        model_path = tmp_path / f"model_{version}.pkl"
        train_and_export_model(df_clean, version=version, model_path=str(model_path))
        assert model_path.exists(), f"El modelo {version} no fue creado correctamente."


def test_model_can_predict(tmp_path):
    """
    Verifica que un modelo entrenado pueda cargar y realizar predicciones válidas.

    Entrena un modelo B, carga sus artefactos y realiza una predicción sobre un
    pasajero simulado, validando la salida.
    """
    # Datos de entrada para predicción
    sample_data = {
        "Pclass": [3],
        "Sex": ["male"],
        "Age": [30.0],
        "SibSp": [0],
        "Parch": [0],
        "Fare": [8.05],
        "Embarked": ["S"],
        "Name": ["Z"],
        "Ticket": ["12345"],
        "Cabin": [None],
        "PassengerId": [999]
    }

    df_sample = pd.DataFrame(sample_data)
    df_clean = preprocess_data(df_sample)

    # Dataset de entrenamiento
    training_data = {
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 2, 3, 2],
        "Sex": ["male", "female", "female", "male", "female"],
        "Age": [22, 38, 26, 35, 27],
        "SibSp": [1, 1, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Fare": [7.25, 71.2833, 7.925, 8.05, 13.00],
        "Embarked": ["S", "C", "S", "S", "Q"],
        "Name": ["A", "B", "C", "D", "E"],
        "Ticket": ["X", "Y", "Z", "K", "L"],
        "Cabin": [None, None, None, None, None],
        "PassengerId": [1, 2, 3, 4, 5]
    }

    df_train = pd.DataFrame(training_data)
    df_train_clean = preprocess_data(df_train)

    model_path = tmp_path / "model_test.pkl"
    train_and_export_model(df_train_clean, version="B", model_path=str(model_path))

    # Cargar modelo y validar predicción
    model_data = joblib.load(model_path)
    model = model_data["model"]
    expected_features = model_data["features"]

    for col in expected_features:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_clean = df_clean[expected_features]

    prediction = model.predict(df_clean)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
