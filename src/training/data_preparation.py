import pandas as pd

def preprocess_data(df):
    """
    Limpia y transforma el dataset del Titanic para entrenamiento de un modelo de clasificación.

    Este preprocesamiento incluye:
        - Eliminación de columnas irrelevantes o con demasiados valores faltantes.
        - Imputación de valores faltantes en 'Age' (mediana) y 'Embarked' (modo = 'S').
        - Codificación one-hot de variables categóricas sin eliminar columnas (drop_first=False).

    Args:
        df (pd.DataFrame): DataFrame original del Titanic.

    Returns:
        pd.DataFrame: Dataset limpio y preparado para entrenamiento.
    """

    # Eliminar columnas irrelevantes o con demasiados nulos
    df = df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, errors='ignore')

    # Imputar valores faltantes en la edad con la mediana
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Imputar valores faltantes en Embarked con la moda ('S')
    df["Embarked"] = df["Embarked"].fillna("S")

    # Codificar variables categóricas sin eliminar ninguna categoría (para consistencia)
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=False)

    return df
