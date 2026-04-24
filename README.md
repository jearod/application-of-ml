# Proyecto Final: Aplicacion del Machine Learning
---
## Tabla de Contenido
- [Resumen](#resumen)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requerimientos](#requerimientos)
- [Conjunto de Datos](#conjunto-de-datos)
- [Pipeline de Entrenamiento: `train_model.py`](#pipeline-de-entrenamiento-train_modelpy)
- [🚀 Cómo ejecutarlo](#-cómo-ejecutarlo)
- [Servidor API: `app.py`](#servidor-api-apppy)
- [⚙️ Configuración de Despliegue](#-configuración-de-despliegue)
- [🐳 Dockerización y Despliegue en Render](#-dockerización-y-despliegue-en-render)
- [🌐 Consumo de la API en Producción](#-consumo-de-la-api-en-producción)
- [Authors](#authors)

## Resumen
Este repositorio contiene el proyecto final para el curso de **Aplicación de Machine Learning**. El proyecto se centra en desarrollar, evaluar y desplegar modelos de machine learning para resolver problemas del mundo real, cubriendo todo el pipeline desde el preprocesamiento de datos hasta la inferencia del modelo.

## Estructura del Proyecto
```
practica-deploy-ml/
├── train_model.py       # Script para entrenar y guardar el modelo
├── app.py               # API de Flask
├── requirements.txt     # Dependencias del proyecto
├── Dockerfile           # Configuración del contenedor Docker
├── .dockerignore        # Archivos a excluir del contenedor (opcional)
├── model.pkl            # Modelo entrenado (se genera al ejecutar train_model.py)
└── README.md            # Descripción del proyecto (opcional)
```

## Requerimientos
Para correr este proyecto localmente asegurate de instalar las siguientes dependencias

```bash
pip install -r requirements.txt
```

## Conjunto de Datos: 

Este proyecto utiliza el clásico **Iris Dataset**, un conjunto de datos estándar en el campo del Machine Learning. Su objetivo es clasificar flores de la planta Iris en tres especies distintas basándose en cuatro medidas morfológicas de sus pétalos y sépalos.

### 📥 Características de Entrada (Features)
El modelo consume 4 variables numéricas (flotantes) que representan las dimensiones físicas de la flor:
| Feature | Descripción |
|---|---|
| `sepal_length` | Longitud del sépalo (cm) |
| `sepal_width` | Ancho del sépalo (cm) |
| `petal_length` | Longitud del pétalo (cm) |
| `petal_width` | Ancho del pétalo (cm) |

### 🎯 Variable Objetivo (Target Classes)
La salida del modelo es un valor entero que mapea a una de las tres especies posibles de la flor Iris:

* **`0`** : *Iris Setosa*
* **`1`** : *Iris Versicolor*
* **`2`** : *Iris Virginica*
---

## Pipeline de Entrenamiento: `train_model.py`

Este script es el responsable de preparar los datos, entrenar los algoritmos de Machine Learning, evaluar su rendimiento y exportar el modelo final que será consumido por la API (`app.py`).

### 🔄 Flujo de Trabajo (Pipeline)

El script sigue un flujo clásico de ciencia de datos dividido en 4 pasos principales:

### 1. Preparación de Datos
* **Dataset:** Carga el conjunto de datos estandarizado **Iris Dataset** directamente desde Scikit-Learn.
* **División (Split):** Separa los datos en dos conjuntos: **70% para entrenamiento** y **30% para pruebas** (`test_size=0.3`). 
* **Estratificación:** Utiliza `stratify=y` para asegurar que la proporción de las tres especies de flores se mantenga equilibrada tanto en el conjunto de entrenamiento como en el de prueba.

### 2. Entrenamiento de Modelos
El script inicializa y entrena dos algoritmos de clasificación diferentes para comparar su rendimiento:
* **SVC (Support Vector Classifier):** Un modelo basado en Máquinas de Vectores de Soporte.
* **Gradient Boosting Classifier (GBC):** Un modelo de ensamble basado en árboles de decisión secuenciales.

### 3. Evaluación de Rendimiento
Para ambos modelos, el script realiza predicciones sobre el conjunto de pruebas (`X_test`) y muestra en la consola de comandos:
* El **Reporte de Clasificación** (`classification_report`): Que incluye precisión (precision), exhaustividad (recall) y el puntaje F1 (f1-score) para cada una de las tres especies.
* El **Puntaje de Exactitud** (`accuracy_score`): El porcentaje total de aciertos redondeado a 3 decimales.

### 4. Exportación (Serialización)
Tras la evaluación, el script toma el modelo **SVC** (elegido como el modelo principal) y lo guarda en el disco duro bajo el nombre de `model.pkl` utilizando la librería `joblib`. Este archivo es el "cerebro" que la API cargará posteriormente para hacer inferencias en tiempo real.

---

## 🚀 Cómo ejecutarlo

Para entrenar el modelo y generar el archivo `.pkl`, simplemente ejecuta el script desde tu terminal:

```bash
python train_model.py
```

---

## Servidor API: `app.py`

Este archivo es el núcleo de la aplicación. Se encarga de exponer un modelo de Machine Learning (entrenado previamente y guardado como `model.pkl`) a través de una **API REST** utilizando el framework **Flask**.

> **Nota técnica:** Aunque el comentario inicial del código menciona "FastAPI", la implementación actual utiliza **Flask**. Flask es ideal para microservicios ligeros y despliegues rápidos de modelos de ciencia de datos.

#### 🛠️ Tecnologías y Librerías
* **Flask:** Framework web para manejar las rutas y peticiones HTTP.
* **Joblib:** Utilizado para deserializar (cargar) el modelo de ML en memoria al arrancar la aplicación.
* **NumPy:** Manejo de arreglos numéricos para dar formato a los datos antes de la predicción.
* **OS:** Para detectar variables de entorno (como el puerto de red) en despliegues en la nube (Render, Railway, etc.).



### 🛣️ Endpoints de la API

#### 1. Información General (`GET /`)
Es el punto de entrada. Proporciona metadatos sobre la API para que el usuario sepa cómo interactuar con ella sin necesidad de leer el código.
* **Respuesta:** Un objeto JSON con la versión, descripción de endpoints y un **ejemplo exacto** de cómo enviar datos al predictor.

#### 2. Predicción de Especies (`POST /predict`)
Es la función principal. Recibe datos de las dimensiones de la flor (iris) y devuelve su clasificación.

**Flujo interno de la función:**
*  **Recepción:** Obtiene el JSON del cuerpo de la petición (`request.get_json()`).
*  **Formateo:** Convierte los valores numéricos (`sepal_length`, `sepal_width`, etc.) en un arreglo bidimensional de NumPy con la forma `(1, 4)`, que es la estructura que el modelo de Scikit-Learn espera.
*  **Inferencia:** Ejecuta `model.predict()` para obtener el índice de la clase y `model.predict_proba()` para obtener la distribución de probabilidades.
*  **Respuesta:** Devuelve un JSON estructurado con:
    * `prediction`: Nombre de la especie (*setosa, versicolor, virginica*).
    * `prediction_index`: El valor entero de la predicción (0, 1 o 2).
    * `probabilities`: Diccionario con el porcentaje de probabilidad para cada especie.
    * `confidence`: El valor máximo de las probabilidades calculadas.

**Manejo de Errores:**
* **400 (Bad Request):** Si falta alguna de las 4 llaves requeridas en el JSON de entrada, es capturado por la excepción `KeyError`.
* **500 (Internal Server Error):** Para cualquier otro fallo inesperado durante la ejecución (capturado por `Exception`).

#### 3. Estado de Salud (`GET /health`)
Un endpoint simple utilizado por servicios de monitoreo, balanceadores de carga o contenedores para verificar que la instancia de la aplicación está activa y respondiendo correctamente.
* **Respuesta:** `{"status": "healthy"}`

---

## ⚙️ Configuración de Despliegue

El bloque final del archivo permite que la aplicación sea flexible según el entorno donde se ejecute:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

**Prioridad de Puerto:** Busca la variable de entorno PORT (utilizada dinámicamente por la mayoría de servicios PaaS o Cloud). Si no está definida, utiliza el puerto 5000 por defecto.

**Host 0.0.0.0:** Esencial para despliegues en contenedores (como Docker) o servidores en la nube, ya que instruye a Flask a escuchar peticiones provenientes de cualquier IP externa, y no solo desde el localhost.

---

## 🐳 Dockerización y Despliegue en Render

La aplicación ha sido empaquetada utilizando **Docker** para garantizar consistencia entre los entornos de desarrollo y producción. Actualmente, la API está desplegada de forma continua y accesible públicamente a través de **Render**.

---

## 🌐 Consumo de la API en Producción

La aplicación se encuentra actualmente desplegada en la nube utilizando **Render** y está disponible para ser consumida públicamente. No necesitas ejecutar el proyecto localmente para probar el modelo de clasificación.

🔗 **URL Base:** - [`https://application-of-ml.onrender.com`](https://application-of-ml.onrender.com)

---

### 📡 Endpoints Disponibles

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Devuelve metadatos, versión y un ejemplo de uso de la API. |
| `GET` | `/health` | Verifica el estado del servidor (`{"status": "healthy"}`). |
| `POST` | `/predict` | **Servicio principal:** Recibe las medidas de la flor y devuelve la predicción. |

---

## 💻 Ejemplos de Integración (`/predict`)

Para realizar una predicción, debes enviar una petición `POST` con un cuerpo en formato JSON que contenga las cuatro medidas requeridas: `sepal_length`, `sepal_width`, `petal_length` y `petal_width`.

### Opción 1: Usando `cURL` (Terminal)
Ideal para pruebas rápidas directamente desde tu línea de comandos:

```bash
curl -X POST [https://application-of-ml.onrender.com/predict](https://application-of-ml.onrender.com/predict) \
     -H "Content-Type: application/json" \
     -d '{
           "sepal_length": 5.1,
           "sepal_width": 3.5,
           "petal_length": 1.4,
           "petal_width": 0.2
         }'
```

### Opción 2: Usando Python (requests)
Ideal si quieres integrar esta API en un notebook de Jupyter, un pipeline de datos o una aplicación externa:

```Python
import requests

url = "[https://application-of-ml.onrender.com/predict](https://application-of-ml.onrender.com/predict)"
payload = {
    "sepal_length": 6.2,
    "sepal_width": 3.4,
    "petal_length": 5.4,
    "petal_width": 2.3
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Predicción exitosa:")
    print(response.json())
else:
    print("Error en la petición:", response.text)
```

### 📥 Formato de Respuesta Esperado
Si la petición es exitosa (Código HTTP 200 OK), la API te devolverá un JSON con la clase predicha, el índice numérico y el desglose de probabilidades para cada especie:

```json
{
  "confidence": 0.89,
  "prediction": "virginica",
  "prediction_index": 2,
  "probabilities": {
    "setosa": 0.02,
    "versicolor": 0.09,
    "virginica": 0.89
  },
  "status": "success"
}
```

> **Nota sobre el rendimiento:** Dado que la API está alojada en el nivel gratuito (Free Tier) de Render, el servidor puede entrar en estado de reposo (sleep) tras un periodo de inactividad. Si tu primera petición tarda unos 30-50 segundos en responder, es normal (arranque en frío). Las peticiones siguientes responderán en milisegundos.

---

## Authors
- **Jean Rodriguez** - *Initial work*