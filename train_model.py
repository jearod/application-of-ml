# ============================================================
# PASO 1: Entrenar y guardar el SVC_modelo
# Archivo: train_SVC_model.py
# ============================================================

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.SVC_model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# -------------------------------------------------------
# TODO 1: Divide los datos en entrenamiento y prueba
# -------------------------------------------------------
# Tu código aquí:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# -------------------------------------------------------
# TODO 2: Elige y entrena un algoritmo de clasificación
# -------------------------------------------------------

# Tu código aquí:
SVC_model = svm.SVC()
SVC_model.fit(X_train,y_train)

GBC_SVC_model = GradientBoostingClassifier()
GBC_SVC_model.fit(X_train,y_train)

# -------------------------------------------------------
# TODO 3: Evalúa el SVC_modelo y muestra métricas
# -------------------------------------------------------

# Tu código aquí:
print("Evaluación del SVC_modelo SVC:")

y_svc_pred = SVC_model.predict(X_test)
print(classification_report(y_test, y_svc_pred, target_names=target_names))
acc_score = round(accuracy_score(y_test, y_svc_pred),3)
print(f"accuracy_score = {acc_score}")

print("Evaluación del SVC_modelo GradientBoostingClassifier:")
y_gbc_pred_pred = GBC_SVC_model.predict(X_test)
print(classification_report(y_test, y_gbc_pred_pred, target_names=target_names))
acc_score2 = round(accuracy_score(y_test, y_gbc_pred_pred),3)
print(f"accuracy_score = {acc_score2}")



# -------------------------------------------------------
# TODO 4: Guarda el SVC_modelo como 'SVC_model.pkl'
# -------------------------------------------------------

# Tu código aquí:
joblib.dump(SVC_model, "model.pkl")
print('¡SVC_modelo SVC guardado correctamente!')