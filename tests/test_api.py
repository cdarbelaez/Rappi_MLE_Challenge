from fastapi.testclient import TestClient
from src.api.main import app

# Cliente de pruebas para simular peticiones HTTP a la API
client = TestClient(app)

# Header con la API Key obligatoria
HEADERS = {"x-api-key": "rappi-secret"}


def test_predict_survival():
    """
    Test unitario del endpoint /predict con un input válido.

    Verifica que la API:
    - Devuelva status code 200.
    - Contenga la clave 'predictions'.
    - Devuelva una lista como resultado.
    """
    sample_passenger = [{
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }]

    response = client.post("/predict", json=sample_passenger, headers=HEADERS)

    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
    assert len(json_data["predictions"]) == 1


def test_predict_with_missing_field():
    """
    Test del endpoint /predict cuando falta un campo obligatorio.

    Verifica que la API responda con error 422 (Unprocessable Entity).
    """
    incomplete_passenger = [{
        "Pclass": 3,
        "Sex": "male",
        # Falta el campo 'Age'
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }]

    response = client.post("/predict", json=incomplete_passenger, headers=HEADERS)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_without_api_key():
    """
    Test que verifica que el endpoint /predict exige API Key.

    Debe retornar error 422 (por header faltante) o 401 (API Key inválida).
    """
    sample_passenger = [{
        "Pclass": 1,
        "Sex": "female",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "C"
    }]

    response = client.post("/predict", json=sample_passenger)  # sin headers
    assert response.status_code in (401, 422)
