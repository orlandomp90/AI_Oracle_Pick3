from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np

from hybrid_cnn_lstm import (
    create_dataset_and_features,
    predict_next_sequence,
    build_hybrid_model,
    WINDOW_SIZE,
    NUM_DIGITS
)

app = Flask(__name__)
# Permitir llamadas desde http://localhost (frontend JS) a este puerto 5001
CORS(app)

MODEL_PATH = "cnn_lstm_weights.weights.h5"
global_model = None

def get_model():
    """Carga el modelo y sus pesos estáticamente en memoria"""
    global global_model
    if global_model is None:
        # El input shape ahora es de 21 features con tamaño de ventana fijo.
        global_model = build_hybrid_model()
        if os.path.exists(MODEL_PATH):
            print(f"✅ Cargando pesos guardados desde {MODEL_PATH}")
            global_model.load_weights(MODEL_PATH)
        else:
            print("⚠️ Modelo virgen inicializado (Sin pesos encontrados). Entrénalo primero.")
    return global_model

@app.route('/api/python/train', methods=['POST'])
def train():
    data = request.json
    if not data or 'draws' not in data:
         return jsonify({"error": "No 'draws' payload provided"}), 400
         
    draws = data['draws']
    if len(draws) <= WINDOW_SIZE:
        return jsonify({"error": f"Se requieren más de {WINDOW_SIZE} sorteos para entrenar."}), 400
        
    print(f"Recibiendo {len(draws)} sorteos para ENTRENAR CNN-LSTM...")
    X, y = create_dataset_and_features(draws, window_size=WINDOW_SIZE)
    
    m = get_model()
    
    # Callback para prevenir memorizar ruido caótico
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("Iniciando fit() asíncrono en background...")
    # Consideración: En un app de prod, esto debería hacerse asíncronamente con Celery o Hilos, 
    # pero para Pick 3 local, unos segundos bloqueando el hilo de Flask son tolerables.
    history = m.fit(
        X, y,
        epochs=30,  # 30 es seguro con early stopping
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1 # Sale por consola Flask
    )
    
    m.save_weights(MODEL_PATH)
    print("Entrenamiento completado y pesos GUARDADOS.")
    
    return jsonify({
        "success": True, 
        "message": f"CNN-LSTM Modelo entrenado exitosamente con loss: {history.history['loss'][-1]:.4f}"
    })

@app.route('/api/python/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'draws' not in data:
         return jsonify({"error": "No 'draws' payload provided"}), 400
         
    draws = data['draws']
    if len(draws) < WINDOW_SIZE:
         return jsonify({"error": f"Faltan datos. Necesitas {WINDOW_SIZE} sorteos exactos de ventana previa."}), 400
    
    print("Recibiendo solicitud de PREDICCIÓN de Frontend...")
    
    # Predecir calculando features sobre TODOS los sorteos para no perder contexto histórico (Ej: g1, g2, g3 y "hot" bugs)
    from hybrid_cnn_lstm import compute_features
    features_scaled = compute_features(draws)
    
    # Solo necesitamos la última ventana de features
    last_window_features = features_scaled[-WINDOW_SIZE:]
    X_pred = np.array([last_window_features])
    m = get_model()
    
    # Monte Carlo Dropout Inference (Simulaciones de Incertidumbre)
    n_samples = 50
    mc_preds_d1, mc_preds_d2, mc_preds_d3 = [], [], []
    
    tensor_x = tf.convert_to_tensor(X_pred)
    
    for _ in range(n_samples):
        # m() llama a la red y el MCDropout siempre aplica el azar iterativo
        p = m(tensor_x)
        mc_preds_d1.append(p[0].numpy())
        mc_preds_d2.append(p[1].numpy())
        mc_preds_d3.append(p[2].numpy())
        
    # Calcular promedios (Voto Consensuado del Transformer)
    mean_d1 = np.mean(mc_preds_d1, axis=0)[0]
    mean_d2 = np.mean(mc_preds_d2, axis=0)[0]
    mean_d3 = np.mean(mc_preds_d3, axis=0)[0]
    
    # Calcular desviación estándar (Incertidumbre/Adivinanza)
    std_d1 = np.std(mc_preds_d1, axis=0)[0]
    std_d2 = np.std(mc_preds_d2, axis=0)[0]
    std_d3 = np.std(mc_preds_d3, axis=0)[0]
    
    mc_means = [mean_d1, mean_d2, mean_d3]
    mc_stds = [std_d1, std_d2, std_d3]

    results = []
    total_entropy = 0.0
    total_uncertainty = 0.0
    
    for i in range(NUM_DIGITS):
        probs = mc_means[i] # Array (10,)
        stds = mc_stds[i]
        
        digit_preds = [
            {
                "digit": int(idx), 
                "probability": float(probs[idx]), 
                "uncertainty": float(stds[idx])
            } 
            for idx in range(10)
        ]
        results.append(digit_preds)
        
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        total_entropy += entropy
        total_uncertainty += float(np.mean(stds))
        
    avg_entropy = total_entropy / 3.0
    
    # Si la varianza promedio (adivinanza empírica) es muy alta, baja la confianza drásticamente
    mc_confidence_penalty = float(total_uncertainty * 2.0)
    
    return jsonify({
        "success": True, 
        "predictions_d1": results[0],
        "predictions_d2": results[1],
        "predictions_d3": results[2],
        "entropy": avg_entropy,
        "mc_uncertainty": total_uncertainty,
        "mc_penalty": mc_confidence_penalty
    })

if __name__ == '__main__':
    # Arrancar en el 5001 en todos los adaptadores
    app.run(host='0.0.0.0', port=5001, debug=False)
