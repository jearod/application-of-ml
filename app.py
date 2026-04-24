# ============================================================
# PASO 2: API de Flask
# Archivo: app.py
# ============================================================

import joblib, os
import numpy as np
from flask import Flask, jsonify, request



# Cargar modelo al arrancar la aplicación
model = joblib.load('model.pkl')

# Clases del dataset Iris (en el mismo orden que sklearn)
CLASSES = ['setosa', 'versicolor', 'virginica']
app = Flask(__name__)

# -------------------------------------------------------
# TODO 5: Implementa el endpoint raíz '/'
# Debe devolver: nombre de la API, versión, algoritmo usado,
# las 4 features esperadas y un ejemplo de llamada a /predict
# -------------------------------------------------------
@app.route('/')
def home():
    return jsonify({
        'message': 'Iris Classification API',
        'version': '1.0',
        'endpoints': {
            '/': 'Info de la API',
            '/predict': 'POST - Clasificar iris',
            '/health': 'Health check'
        },
        'example': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'sepal_length': 5.1,
                'sepal_width': 3.5,
                'petal_length': 1.4,
                'petal_width': 0.2
            }
        }
    })


# -------------------------------------------------------
# TODO 6: Implementa el endpoint '/predict'
# - Devuelve prediction, probabilities y status
# - Maneja errores: campos faltantes (KeyError) y otros (Exception)
# -------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extraer features
        features = np.array([
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]).reshape(1, -1)
        
        # Predicción
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': CLASSES[prediction],
            'prediction_index': int(prediction),
            'probabilities': {
                species: round(float(prob),2)
                for species, prob in zip(CLASSES, probabilities)
            },
            'confidence': round(float(max(probabilities)),2),
            'status': 'success'
        })
    
    except KeyError as e:
        return jsonify({
            'error': f'Campo requerido faltante: {str(e)}',
            'status': 'error'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500



# -------------------------------------------------------
# TODO 7: Implementa el endpoint '/health'
# Debe devolver {"status": "healthy"} con código 200
# -------------------------------------------------------
@app.route('/health')
def health():
    # Tu código aquí:
    return jsonify ({'status': 'healthy'})


if __name__ == '__main__':
    # Render/Railway asigna el puerto mediante la variable PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)