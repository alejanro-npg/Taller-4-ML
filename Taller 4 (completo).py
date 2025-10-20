# %% [markdown]
# ### PCA in Machine Learning Workflows
# #### Machine Learning I - Maestría en Analítica Aplicada
# #### Universidad de la Sabana
# #### Prof: Hugo Franco
# #### Exercise: Dealing with Class Imbalance II

# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, make_scorer, f1_score, mean_squared_error, r2_score, mean_absolute_error
from prefect import flow, task, get_run_logger
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score    
from scipy.stats import randint, uniform
import time
from urllib.parse import urlparse




# %% [markdown]
# 1. Data Loading
# Start by importing the libraries and loading the dataset and describe its contents

# %%
@task(name="Load Dataset")
def load_dataset(flow_name: str, path: str = None):
    """
    Carga el dataset correspondiente según el flujo que se esté ejecutando.
    Si no se especifica `path`, se asigna uno por defecto según el nombre del flujo.

    Parámetros:
    -----------
    flow_name : str
        Nombre del flujo ('credit_card', 'customer_defection', 'bike_rental')
    path : str, opcional
        Ruta local o URL del dataset. Si no se da, usa la predeterminada según el flujo.

    Retorna:
    --------
    pd.DataFrame : dataset cargado en memoria
    """

    # 🔹 Asignar rutas por defecto
    default_paths = {
        "credit_card": "creditcard.csv",
        "customer_defection": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "bike_rental": "day.csv"
    }

    # 🔹 Determinar la fuente del dataset
    if path is None:
        if flow_name in default_paths:
            path = default_paths[flow_name]
        else:
            raise ValueError(f"❌ No hay un dataset configurado para el flujo '{flow_name}'")

    # 🔹 Detectar si `path` es URL o ruta local
    is_url = urlparse(path).scheme in ("http", "https")

    try:
        if is_url:
            df = pd.read_csv(path)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ El archivo local '{path}' no existe.")
            df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar el dataset desde '{path}': {e}")

    # 🔹 Información del dataset
    print(f"✅ [{flow_name.upper()}] Dataset cargado correctamente:")
    print(f"   → Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

    # 🔹 Mensaje descriptivo según flujo
    messages = {
        "credit_card": "💳 Dataset de detección de fraude cargado (transacciones con etiqueta de fraude).",
        "customer_defection": "📞 Dataset de customer churn cargado (clientes que abandonan o permanecen).",
        "bike_rental": "🚴 Dataset de alquiler de bicicletas cargado (valores numéricos para regresión)."
    }
    print(messages.get(flow_name, "⚠️ Flujo desconocido. Dataset cargado sin descripción específica."))

    return df


# %%
@task(name="Analyze Dataset")
def data_analysis(flow_name: str,df: pd.DataFrame):
    if flow_name == "bike_rental":
        print("📊 Análisis exploratorio del dataset:\n")
        print(df.info())
        print("\nEstadísticas descriptivas:\n", df.describe())
        print("\nValores nulos por columna:\n", df.isnull().sum())

        plt.figure(figsize=(8,4))
        sns.histplot(df['cnt'], bins=30, kde=True, color='skyblue')
        plt.title('Distribución de la variable cnt (renta total de bicicletas)')
        plt.xlabel('cnt')
        plt.ylabel('Frecuencia')
        plt.show()

        plt.figure(figsize=(10,6))
        sns.heatmap(df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr(), annot=True, cmap='coolwarm')
        plt.title('Matriz de correlación')
        plt.show()

        sns.boxplot(x='season', y='cnt', data=df)
        plt.title('Rentas según la temporada')
        plt.xlabel('Temporada (1: Invierno, 2: Primavera, 3: Verano, 4: Otoño)')
        plt.ylabel('Total de rentas')
        plt.show()
    
    elif flow_name == "credit_card":
            print (df.info())
            # Check class distribution
            print("Class Distribution:")
            print(df['Class'].value_counts())
            print("\nPercentage Distribution:")
            print(df['Class'].value_counts(normalize=True) * 100)
    
    elif flow_name == "customer_defection":
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

# %%
# Scale 'Time' and 'Amount' features

@task(name="Data preparation")
def data_preparation(df):
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Define features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split Early, Split Once: Separate into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# %%
@task(name="Preprocess Data")
def preprocess_data(flow_name: str,df: pd.DataFrame):
    if flow_name == "bike_rental":
        print("⚙️ Iniciando preprocesamiento de datos...")

        # Conversión de fecha y creación de nuevas columnas
        df['dteday'] = pd.to_datetime(df['dteday'])
        df['day'] = df['dteday'].dt.day
        df['month'] = df['dteday'].dt.month
        df['year'] = df['dteday'].dt.year
        df['weekday'] = df['dteday'].dt.weekday

        # Eliminación de columnas irrelevantes
        df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

        # Codificación de variables categóricas
        df = pd.get_dummies(df, columns=['season', 'weathersit'], drop_first=True)

        # División entre features y target
        X = df.drop('cnt', axis=1)
        y = df['cnt']

        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Escalado de variables numéricas
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✅ Preprocesamiento completado correctamente para Bike Rental.")
        return X_train, X_test, y_train, y_test, scaler
        
    elif flow_name == "customer_defection":
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
        print("✅ Preprocesamiento completado correctamente.")
        return X_train, X_test, y_train, y_test, preprocessor

# %% [markdown]
# 2. Data Preprocessing
# 
# Necessary preprocessing steps to clean the data and prepare it for the pipeline.

# %%
@task(name="Model Selection")
def model_selection(X_train, y_train, flow_type="default", preprocessor=None):
    """
    Selección de modelo adaptable a diferentes flujos:
      - Clasificación binaria (customer_defection, credit_card)
      - Regresión (bike_rental)
    """

    start_time = time.time()
    print(f"🔍 Iniciando selección de modelo para flujo: {flow_type}\n")

    # 🔹 Usar una muestra más pequeña para acelerar búsquedas
    if len(X_train) > 5000:
        X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42, stratify=y_train if flow_type != "bike_rental" else None)
    else:
        X_sample, y_sample = X_train, y_train

    # 🔹 Clasificación binaria (customer_defection o credit_card)
    if flow_type in ["customer_defection", "credit_card"]:
        models = {
            "XGBoost": XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42
            ),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
        }
        scoring = make_scorer(f1_score)
        smote_step = ('smote', SMOTE(random_state=42))
    
    # 🔹 Regresión (bike_rental)
    elif flow_type == "bike_rental":
        models = {
            "XGBoostRegressor": XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            ),
            "LinearRegression": LinearRegression()
        }
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        smote_step = None  # No se usa SMOTE en regresión
    
    else:
        raise ValueError(f"❌ Tipo de flujo no reconocido: {flow_type}")

    best_model = None
    best_score = -np.inf
    best_name = ""

    # 🔹 Iterar sobre cada modelo
    for name, model in models.items():
        steps = []

        # ⚙️ Solo agregar preprocessor si aplica (no para credit_card)
        if preprocessor is not None and flow_type != "credit_card":
            steps.append(('preprocessor', preprocessor))

        # ⚙️ Agregar paso SMOTE si existe
        if smote_step is not None:
            steps.append(smote_step)

        # ⚙️ Paso del modelo
        steps.append(('model', model))

        pipeline = Pipeline(steps)

        # 🔹 Búsqueda de hiperparámetros solo para XGBoost
        if "XGBoost" in name:
            param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=3,
                scoring=scoring,
                cv=2,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            search.fit(X_sample, y_sample)
            score = search.best_score_
            model_used = search.best_estimator_

        else:
            pipeline.fit(X_sample, y_sample)

            if flow_type == "bike_rental":
                preds = pipeline.predict(X_sample)
                score = -mean_squared_error(y_sample, preds)  # Negativo porque MSE penaliza al revés
            else:
                score = f1_score(y_sample, pipeline.predict(X_sample))

            model_used = pipeline

        print(f" → {name}: Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model_used
            best_name = name

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ Mejor modelo: {best_name} con score = {best_score:.4f}")
    print(f"⏱️ Tiempo total de selección: {elapsed:.2f} min\n")


    return best_name, best_model


# %% [markdown]
# We build and train the model

# %%
@task(name="Build and Train the model")
def build_and_train_pipeline(best_model, X_train, y_train):
    print("  Entrenando el mejor modelo seleccionado...")
    best_model.fit(X_train, y_train)
    print("  Entrenamiento completado con éxito.")
    return best_model


# %% [markdown]
# 4. Evaluate the Model
# Finally, we use the trained pipeline to make predictions on the untouched test set and evaluate its performance.

# %%
@task(name="Evaluate Model")
def evaluate_model(flow_name: str, pipeline, X_test, y_test):
    print(f"\n🚀 Evaluando el modelo para el flujo: {flow_name.upper()}...\n")

    # Predicciones del modelo
    y_pred = pipeline.predict(X_test)

    # --- 🔹 Flujos de CLASIFICACIÓN ---
    if flow_name in ["credit_card", "customer_defection"]:
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        print("📊 Classification Report:\n")
        report = classification_report(
            y_test,
            y_pred,
            target_names=['Clase Negativa', 'Clase Positiva']
        )
        print(report)

        roc_auc = None
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"🎯 ROC AUC Score: {roc_auc:.3f}\n")

        print("✅ Evaluación de clasificación completada exitosamente.\n")

        return {
            "classification_report": report,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

    # --- 🔹 Flujo de REGRESIÓN ---
    elif flow_name == "bike_rental":
        print("\n📈 Generando métricas de regresión...\n")
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  • MSE  = {mse:.4f}")
        print(f"  • MAE  = {mae:.4f}")
        print(f"  • R²   = {r2:.4f}")

        print("\n📊 Generando regression plot...\n")

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "y_pred": y_pred
        }

    else:
        raise ValueError(f"❌ Flujo '{flow_name}' no reconocido para evaluación.")


# -------------------------------------------------------------------
# MATRICES DE CONFUSIÓN
# -------------------------------------------------------------------

def plot_confusion_matrices(flow_name, y_test, y_pred, output_dir="./artifacts"):
    # Crear carpeta si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("\nGenerating plots...")
    cm_row = confusion_matrix(y_test, y_pred, normalize='true')
    cm_all = confusion_matrix(y_test, y_pred, normalize='all')
    cm_pred = confusion_matrix(y_test, y_pred, normalize='pred')

    class_names = (
        ['Not Fraud', 'Fraud'] if flow_name == "credit_card" else ['Not Churn', 'Churn']
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrices = [cm_row, cm_all, cm_pred]
    titles = ["Row-Normalized", "Global-Normalized", "Column-Normalized"]

    for ax, cm, title in zip(axes, matrices, titles):
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax)
        ax.set_title(f'Confusion Matrix ({title})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names, rotation=0)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{flow_name}_confusion_matrices.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()

    print(f"✅ Confusion matrices saved in {save_path}")

# -------------------------------------------------------------------
# PLOT REGRESIÓN PARA BIKE RENTAL
# -------------------------------------------------------------------

def plot_regression_results(y_test, y_pred, output_dir="./artifacts"):
       # Crear carpeta si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("\nGenerating plots...")
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Bike Rental)")

    save_path = os.path.join(output_dir, "bike_rental_regression_plot.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()

    print(f"✅ Regression plot saved in {save_path}")


# %%
def run_full_flow(flow_name: str):
    """
    Ejecuta el flujo completo dinámicamente según el tipo de dataset:
    - "credit_card": Clasificación binaria (fraude)
    - "customer_defection": Clasificación binaria (churn)
    - "bike_rental": Regresión (demanda de bicicletas)
    """

    print(f"\n🚀 Iniciando flujo: {flow_name.upper()}\n")

    # -----------------------------
    # 1️⃣ Carga del dataset
    # -----------------------------
    print("🔹 Paso 1: Cargando los datos...")
    df = load_dataset(flow_name)

    # -----------------------------
    # 2️⃣ Análisis exploratorio
    # -----------------------------
    print("🔹 Paso 2: Análisis exploratorio del dataset...")
    data_analysis(flow_name, df)

    # -----------------------------
    # 3️⃣ Preprocesamiento y división
    # -----------------------------
    print("🔹 Paso 3: Preprocesamiento y división de datos...")
    if flow_name in ["bike_rental", "customer_defection"]:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(flow_name, df)
    elif flow_name == "credit_card":
        X_train, X_test, y_train, y_test = data_preparation(df)

    # -----------------------------
    # 4️⃣ Selección de modelo
    # -----------------------------
    print("🔹 Paso 4: Selección de modelos...")
    if flow_name in ["bike_rental", "customer_defection"]:
        best_model_name, best_model = model_selection( X_train, y_train,flow_name, preprocessor)
    elif flow_name == "credit_card":
        best_model_name, best_model = model_selection( X_train, y_train,flow_name)

    # -----------------------------
    # 5️⃣ Entrenamiento final
    # -----------------------------
    print(f"🔹 Paso 5: Entrenamiento final del modelo {best_model_name}...")
    trained_model = build_and_train_pipeline(best_model, X_train, y_train)

    # -----------------------------
    # 6️⃣ Evaluación
    # -----------------------------
    print("🔹 Paso 6: Evaluación del modelo...")
    results = evaluate_model(flow_name, trained_model, X_test, y_test)
    y_pred = results.get("y_pred")

    # -----------------------------
    # 7️⃣ Visualización
    # -----------------------------
    print("🔹 Paso 7: Visualización de resultados...")
    output_dir = "./artifacts"  
    if flow_name in ["credit_card", "customer_defection"]:  
        plot_confusion_matrices(flow_name, y_test, y_pred, output_dir)
    elif flow_name == "bike_rental":
        plot_regression_results(y_test, y_pred, output_dir)

    else:
        print("⚠️ No se definió una visualización para este flujo.")

    print(f"\n✅ Flujo completo '{flow_name.upper()}' ejecutado exitosamente.\n")


# %%
if __name__ == "__main__":
    print("\n [CREDIT CARD FRAUD DETECTION FLOW] Iniciando flujo de trabajo para la clasificación de fraudes con tarjeta de crédito...\n")
    run_full_flow("credit_card")
    print("\n [Customer Defection Flow] Iniciando flujo de trabajo para la clasificación de deserción de clientes...\n")
    run_full_flow("customer_defection")
    print("\n [Bike Rental Hours FLOW] Iniciando flujo de trabajo para la regresión para el conteo horario total de renta de bicicletas...\n")
    run_full_flow("bike_rental")


