# %% [markdown]
# ### PCA in Machine Learning Workflows
# #### Machine Learning I - Maestría en Analítica Aplicada
# #### Universidad de la Sabana
# #### Prof: Hugo Franco
# #### Exercise: Dealing with Class Imbalance II

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from prefect import flow, task, get_run_logger


# %% [markdown]
# 1. Data Loading
# Start by importing the libraries and loading the dataset and describe its contents

# %%
@task(name="Load Dataset")
def load_dataset():
    """
    Carga el dataset de churn desde el repositorio público de IBM.
    
    Returns:
        df (DataFrame): DataFrame con los datos cargados.
    """
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    print("Dataset cargado correctamente")
    return df

# %%

@task(name="Analyze Data")
def analyze_dataset(df):
    """
    Realiza un análisis exploratorio inicial del dataset:
    - Tamaño y estructura
    - Tipos de variables
    - Distribución de la clase objetivo (Churn)
    - Estadísticas descriptivas
    - Valores únicos en variables categóricas
    - Visualización de la variable objetivo
    - Valores faltantes
    
    Args:
        df (DataFrame): Dataset cargado
    
    Returns:
        dict: Resumen del análisis con features numéricos y categóricos
    """
    print(" Dataset Shape:", df.shape)

    # Identificar tipos de variables
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

    # Info general
    print("\n Dataset Info:")
    print(df.info())

    # Distribución de la variable objetivo
    print("\n Churn Distribution:")
    print(df['Churn'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    # Estadísticas descriptivas
    if numerical_features:
        print("\n Numerical Features Summary:")
        print(df[numerical_features].describe())
    else:
        print("\n No numerical features available before preprocessing.")

    # Valores únicos de las categóricas
    if categorical_features:
        print("\n Categorical Features Overview:")
        for col in categorical_features:
            print(f"\n{col} unique values:")
            print(df[col].value_counts())
    else:
        print("\n No categorical features available before preprocessing.")

    # Visualización distribución churn
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Churn')
    plt.title('Distribution of Customer Churn')
    plt.xlabel('Churn (No / Yes)')
    plt.ylabel('Count')
    plt.show()

    # Valores faltantes
    print("\n Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0]) if missing_values.any() else print("No missing values detected.")

    # Retornar resumen para posible uso posterior
    return {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features
    }

# %% [markdown]
# 2. Data Preprocessing
# 
# Necessary preprocessing steps to clean the data and prepare it for the pipeline.

# %%

@task(name="Preprocess data")
def preprocess_data(df):
    """
    Realiza el preprocesamiento completo del dataset Telco Customer Churn.
    Incluye limpieza, transformación de variables y división del dataset.

    Pasos:
    1. Elimina 'customerID' por ser irrelevante.
    2. Convierte 'TotalCharges' a numérico y elimina filas con NaN.
    3. Convierte 'Churn' y 'gender' en variables binarias.
    4. Separa las variables predictoras (X) de la variable objetivo (y).
    5. Identifica variables numéricas y categóricas.
    6. Crea un pipeline de preprocesamiento con escalado y one-hot encoding.
    7. Divide el dataset en entrenamiento y prueba.

    Args:
        df (DataFrame): Dataset original cargado.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # 1. Eliminar identificador
    df = df.drop('customerID', axis=1)

    # 2. Convertir columna TotalCharges a numérica
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # 3. Convertir variables binarias
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # 4. Separar X e y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 5. Identificar variables numéricas y categóricas
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # 6. Crear pipeline de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Mantener columnas no procesadas
    )

    # 7. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(" Preprocesamiento completado.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor

# %% [markdown]
# 3. Build and Train the XGBoost Pipeline
# Here, we define our three-step pipeline. When we call `.fit()`, it will automatically preprocess the data, apply SMOTE to the results, and then train the XGBoost classifier on the balanced data.

# %%
@task(name="Build and Train the XGBoost Pipeline")
def build_and_train_pipeline(preprocessor, X_train, y_train):  
    # Crear pipeline completo
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        ))
    ])

    # Entrenar pipeline
    print(" Entrenando pipeline con SMOTE + XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    print(" Entrenamiento completado con éxito.")

    return xgb_pipeline

# %% [markdown]
# 4. Evaluate the Model
# Finally, we use the trained pipeline to make predictions on the untouched test set and evaluate its performance.

# %%
@task(name="Evaluate model")
def evaluate_model(xgb_pipeline, X_test, y_test):
    print(" Evaluando el modelo en el conjunto de prueba...\n")

    # 1. Predicciones
    y_pred = xgb_pipeline.predict(X_test)
    y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    # 2. Reporte de clasificación
    print(" Classification Report:\n")
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    print(report)

    # 3. Calcular ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f" ROC AUC Score: {roc_auc:.3f}")

    print("\n Evaluación completada.")

    # Retornar métricas útiles para registro o visualización adicional
    return {
        "classification_report": report,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

# %%
@task(name="Confusion matrix")
def plot_confusion_matrices(y_test, y_pred):
    # Calcular matrices
    cm_row = confusion_matrix(y_test, y_pred, normalize='true')
    cm_all = confusion_matrix(y_test, y_pred, normalize='all')
    cm_pred = confusion_matrix(y_test, y_pred, normalize='pred')

    # Crear figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Matriz normalizada por filas (True) ---
    sns.heatmap(cm_row, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Row-Normalized)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # --- Matriz normalizada globalmente (All) ---
    sns.heatmap(cm_all, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Global-Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # --- Matriz normalizada por columnas (Pred) ---
    sns.heatmap(cm_pred, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=axes[2])
    axes[2].set_title('Confusion Matrix (Column-Normalized)')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')

    # Etiquetas de clases
    class_names = ['Not Churn', 'Churn']
    for ax in axes:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names, rotation=0)

    plt.tight_layout()
    plt.show()

    print(" Confusion matrices plotted successfully.")

    ### Conclusión sobre las matrices de confusión
# De las tres matrices mostradas, **la mejor interpretación proviene de la matriz normalizada por filas (Row-Normalized)**.  
# 
# Esto se debe a que en este tipo de normalización cada fila representa una clase real, y los valores muestran la proporción 
# de predicciones correctas e incorrectas respecto al total de casos reales de esa clase.  
# 
# Por ejemplo, se observa que:
# - El **85 % de los clientes que realmente no hacen churn** fueron correctamente clasificados como *Not Churn*.
# - El **60 % de los clientes que realmente hacen churn** fueron correctamente clasificados como *Churn*.
# 
# Esto permite analizar fácilmente **la capacidad del modelo para reconocer correctamente cada clase**, identificando 
# desequilibrios o debilidades.  
# En cambio, las normalizaciones global y por columnas pueden distorsionar esta interpretación al mezclar las proporciones 
# entre clases o respecto a las predicciones.
# 
# **En conclusión**, la matriz **Row-Normalized** es la más adecuada para evaluar el rendimiento del modelo de clasificación, 
# ya que muestra de forma clara y directa **qué tan bien predice cada clase en relación con los datos reales.**

    return {
        "cm_row": cm_row,
        "cm_all": cm_all,
        "cm_pred": cm_pred
    }


# %%
def run_full_flow():
    print("🔹 Paso 1: Cargando los datos...")
    df = load_dataset()

    print("🔹 Paso 2: Análisis exploratorio del dataset...")
    analyze_dataset(df)

    print("🔹 Paso 3: Preprocesamiento y división de datos...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("🔹 Paso 4: Entrenamiento del modelo con SMOTE + XGBoost...")
    xgb_pipeline = build_and_train_pipeline(preprocessor, X_train, y_train)

    print("🔹 Paso 5: Evaluación del modelo...")
    results = evaluate_model(xgb_pipeline, X_test, y_test)
    y_pred = results["y_pred"]
    y_proba = results["y_proba"]

    print("🔹 Paso 6: Visualización de matrices de confusión...")
    plot_confusion_matrices(y_test, y_pred)

    print("✅ Flujo completo ejecutado exitosamente.")


# %%
if __name__ == "__main__":
    run_full_flow()


