import joblib
import pandas as pd
from utils import kaggle_format
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Carga de datos
data = pd.read_parquet('logs/data_train.parquet')
X_kaggle = pd.read_parquet('logs/data_test.parquet')
X = data.drop(columns=["var_rpta_alt"])
y = data["var_rpta_alt"]

# División estratificada de datos
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Configuración de modelos
models = [
    # {
    #     'name': 'LogisticRegression',
    #     'pipeline': Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('model', LogisticRegression(max_iter=1000, random_state=42))
    #     ]),
    #     'params': {
    #         'model__C': [10, 20],
    #         'model__penalty': ['l1', 'l2'],
    #         'model__solver': ['liblinear']
    #     }
    # },
    {
        'name': 'RandomForest',
        'pipeline': Pipeline([
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'params': {
            'model__n_estimators': [300, 500],
            'model__max_depth': [30, 50],
            'model__class_weight': ['balanced']
        }
    },
    # {
    #     'name': 'SVM',
    #     'pipeline': Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('model', SVC(random_state=42, probability=True))
    #     ]),
    #     'params': {
    #         'model__C': [0.1, 1, 10],
    #         'model__kernel': ['linear', 'rbf']
    #     }
    # },
    # {
    #     'name': 'XGBoost',
    #     'pipeline': Pipeline([
    #         ('model', XGBClassifier(
    #             random_state=42,
    #             eval_metric='logloss',
    #             use_label_encoder=False
    #         ))
    #     ]),
    #     'params': {
    #         'model__learning_rate': [0.05, 0.1],
    #         'model__n_estimators': [100, 200],
    #     }
    # }
]

best_score = 0
best_model = None

# Búsqueda del mejor modelo
for config in models:
    print(f"\n{'=' * 50}")
    print(f"Entrenando {config['name']}...")

    grid = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['params'],
        scoring='f1',
        cv=3,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # Evaluación en validación
    val_pred = grid.predict(X_val)
    f1_val = f1_score(y_val, val_pred)

    print(f"\nMejores parámetros: {grid.best_params_}")
    print(f"F1 CV: {grid.best_score_:.4f} | F1 Validación: {f1_val:.4f}")

    if f1_val > best_score:
        best_score = f1_val
        best_model = grid.best_estimator_
        print(f"¡Nuevo mejor modelo: {config['name']}!")

# Entrenamiento final con todos los datos
print("\nEntrenando modelo final con train + validation...")
X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])
best_model.fit(X_full_train, y_full_train)

# Evaluación en test
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1]

print("\nEvaluación final en test:")
print(f"F1-Score: {f1_score(y_test, test_pred):.4f}")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, test_pred))

# Guardado del modelo
joblib.dump(best_model, 'logs/best_model.pkl')
print("\nModelo guardado como 'best_model.pkl'")

# Predicción para Kaggle
y_kaggle_pred = best_model.predict(X_kaggle)
y_kaggle_proba = best_model.predict_proba(X_kaggle)[:, 1]

sub = pd.read_csv('data/test/prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv')

# Guardar predicciones binarias
kaggle_format(sub, y_kaggle_pred)

# Guardar probabilidades
sub_proba = sub.copy()
sub_proba['Prob_uno'] = y_kaggle_proba
sub_proba.to_csv('logs/submission_probas.csv', index=False)

print("\nArchivos generados:")
print("- submission_final.csv (predicciones binarias)")
print("- submission_probas.csv (probabilidades clase positiva)")