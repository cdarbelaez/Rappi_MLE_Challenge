from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def train_and_export_model(df, version="A", model_path=None):
    """
    Entrena y exporta un modelo de clasificación binaria (versión A o B).

    La versión A usa una regresión logística simple.
    La versión B utiliza un modelo mejorado con escalado y Gradient Boosting.

    Args:
        df (pd.DataFrame): Dataset con features y la columna 'Survived'.
        version (str): Identificador del modelo ('A' o 'B').
        model_path (str, optional): Ruta personalizada para guardar el modelo.

    Returns:
        None
    """
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    if version == "A":
        model = LogisticRegression(max_iter=1000, random_state=42)

    elif version == "B":
        # Modelo B: Pipeline con ColumnTransformer + Gradient Boosting
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features)
            ],
            remainder="passthrough"
        )

        model = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

    else:
        raise ValueError("Versión de modelo no reconocida: usa 'A' o 'B'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== Resultados del Modelo {version} ===")
    print("Features usadas:", X_train.columns.tolist())
    print(classification_report(y_test, y_pred))

    # Definir ruta por defecto si no se proporciona
    if model_path is None:
        model_path = f"models/model_{version}.pkl"

    joblib.dump({
        "model": model,
        "features": X_train.columns.tolist()
    }, model_path)

    print(f"Modelo {version} exportado exitosamente a {model_path}")
