import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
import datetime
import math
import re
import time
import random
import os

# CONFIGURACION DE SEMILLA DINAMICA (ALEATORIZADA)
# Se genera una diferente en cada ejecución basándose en el reloj del sistema.
# Esto asegura diversidad en las redes neuronales para hallar patrones ocultos en el Pick 3.
SEED = int(time.time())
print(f"[IA ORACLE] Inicializando motor neuronal con semilla dinámica: {SEED}")

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 30
NUM_DIGITS = 3
DIGIT_CLASSES = 10 # 0 al 9
NUM_FEATURES = 21

def parse_date(date_str):
    shift = 0.5
    if not isinstance(date_str, str):
        date_str = str(date_str)
        
    l = date_str.lower()
    if 'mid' in l or 'md' in l or 'day' in l:
        shift = 0.0
    elif 'eve' in l or 'night' in l or 'pm' in l:
        shift = 1.0
        
    try:
        from dateutil import parser
        d = parser.parse(date_str, fuzzy=True)
    except:
        match = re.search(r'(\w{3})\s+(\d{1,2}),\s+(\d{4})', date_str)
        if match:
            # Format like 'Mar 30, 2026'
            try:
                d = datetime.datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%b %d %Y")
            except:
                d = datetime.datetime.now()
        else:
            match2 = re.search(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', date_str)
            if match2:
                m, day, y = int(match2.group(1)), int(match2.group(2)), int(match2.group(3))
                if y < 100: y += 2000
                try:
                    d = datetime.datetime(y, m, day)
                except:
                    d = datetime.datetime.now()
            else:
                d = datetime.datetime.now()
            
    dow = d.weekday() + 1 # Mon=1, Sun=7
    if dow == 7: dow = 0 # Sun=0
    return (dow / 6.0, d.month / 12.0, d.day / 31.0, dow, shift)

def compute_features(draws_series):
    """
    draws_series es una lista de listas o tuplas, ej: [[5,0,6, 'Mar 30, 2026'], ...]
    Extraemos 21 features por cada sorteo.
    """
    features = []
    
    for i in range(len(draws_series)):
        row = draws_series[i]
        d1, d2, d3 = int(row[0]), int(row[1]), int(row[2])
        date_str = row[3] if len(row) > 3 else ""
        
        dow, mon, dom, dayNum, shift = parse_date(date_str)
        
        sum_val = (d1 + d2 + d3) / 27.0
        has_repeat = 1.0 if (d1 == d2 or d2 == d3 or d1 == d3) else 0.0
        
        g1, g2, g3 = 50, 50, 50
        for j in range(i - 1, max(-1, i - 51), -1):
            if g1 == 50 and draws_series[j][0] == d1: g1 = i - j
            if g2 == 50 and draws_series[j][1] == d2: g2 = i - j
            if g3 == 50 and draws_series[j][2] == d3: g3 = i - j
            
        s5 = 0.0
        for j in range(max(0, i - 5), i):
            s5 += draws_series[j][0] + draws_series[j][1] + draws_series[j][2]
        s5 /= 135.0
        
        tN1, tN2, tN3, tw = 0.0, 0.0, 0.0, 0.0
        for j in range(max(0, i - 5), i):
            tN1 += draws_series[j][0]
            tN2 += draws_series[j][1]
            tN3 += draws_series[j][2]
            tw += 1
        tw = max(tw, 1.0)
        
        even = sum(1 for n in (d1, d2, d3) if n % 2 == 0) / 3.0
        r_val = (max(d1, d2, d3) - min(d1, d2, d3)) / 9.0
        isWknd = 1.0 if (dayNum == 0 or dayNum == 6) else 0.0
        
        entropy = 0.5
        if i >= 20:
            f = [0] * 10
            for j in range(i - 20, i):
                f[int(draws_series[j][0])] += 1
                f[int(draws_series[j][1])] += 1
                f[int(draws_series[j][2])] += 1
            entropy = 0.0
            for v in f:
                if v > 0:
                    p = v / 60.0
                    entropy -= p * math.log2(p)
            entropy /= math.log2(10)
            
        hot = 0.0
        hw = min(30, i)
        for j in range(max(0, i - hw), i):
            w = (j - max(0, i - hw)) / max(hw, 1.0)
            if draws_series[j][0] == d1: hot += w
            if draws_series[j][1] == d2: hot += w
            if draws_series[j][2] == d3: hot += w
            
        feat = [
            d1/9.0, d2/9.0, d3/9.0,
            dow, mon, dom,
            shift, sum_val, has_repeat,
            min(g1, 50)/50.0, min(g2, 50)/50.0, min(g3, 50)/50.0,
            s5,
            tN1/(tw*9.0), tN2/(tw*9.0), tN3/(tw*9.0),
            even, r_val, isWknd,
            entropy, min(hot, 3.0)/3.0
        ]
        features.append(feat)
        
    return np.array(features, dtype=float)

def create_dataset_and_features(draws_series, window_size=WINDOW_SIZE):
    features_scaled = compute_features(draws_series)
    
    X, y1, y2, y3 = [], [], [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i : i + window_size])
        
        target_d1, target_d2, target_d3 = draws_series[i + window_size][0:3]
        y1.append(int(target_d1))
        y2.append(int(target_d2))
        y3.append(int(target_d3))
        
    return np.array(X), [np.array(y1), np.array(y2), np.array(y3)]

def predict_next_sequence(last_window_draws):
    features_scaled = compute_features(last_window_draws)
    # Shape devuelto: (1, 30, 21)
    return np.array([features_scaled])

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = MCDropout(dropout)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = MCDropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_hybrid_model(input_shape=(WINDOW_SIZE, NUM_FEATURES)):
    inputs = Input(shape=input_shape)
    
    # 1. Feature Extractor (Local awareness)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    
    # 2. Transformer Blocks (Self-Attention Layer)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.25)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.25)
    
    # 3. Global Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # 4. Dense Network with Monte Carlo Dropout (Atención a la incertidumbre)
    x = MCDropout(0.3)(x)
    shared_dense = Dense(64, activation='relu')(x)
    shared_dense = MCDropout(0.2)(shared_dense)
    
    out_d1 = Dense(DIGIT_CLASSES, activation='softmax', name='digit_1')(shared_dense)
    out_d2 = Dense(DIGIT_CLASSES, activation='softmax', name='digit_2')(shared_dense)
    out_d3 = Dense(DIGIT_CLASSES, activation='softmax', name='digit_3')(shared_dense)
    
    model = Model(inputs=inputs, outputs=[out_d1, out_d2, out_d3])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'digit_1': 'sparse_categorical_crossentropy',
            'digit_2': 'sparse_categorical_crossentropy',
            'digit_3': 'sparse_categorical_crossentropy'
        },
        metrics={
            'digit_1': 'accuracy',
            'digit_2': 'accuracy',
            'digit_3': 'accuracy'
        }
    )
    
    return model
