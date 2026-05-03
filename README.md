<div align="center">
  <h1>🧠 AI Oracle Pick3</h1>
  <p><em>Advanced Hybrid Neural Network (CNN + Transformer) for Pick 3 Lottery Prediction</em></p>
</div>

---

## 📖 Descripción del Proyecto

**AI Oracle Pick3** es un oráculo predictivo avanzado impulsado por Inteligencia Artificial, diseñado específicamente para analizar, modelar y predecir los resultados del sorteo tipo "Pick 3". A diferencia de métodos estadísticos tradicionales, este sistema emplea arquitecturas de Deep Learning de última generación combinando **Redes Neuronales Convolucionales (CNN)** y **Bloques Transformer (Self-Attention)**.

El sistema no solo genera predicciones, sino que también evalúa la **incertidumbre estocástica** mediante simulaciones de **Monte Carlo Dropout**, dándole al usuario una medida de confianza ("entropía") sobre cada predicción en función del comportamiento aleatorio de la red.

## 🏗️ Arquitectura del Sistema

El proyecto está dividido en dos capas principales:

1. **Backend Predictivo (Python)**
   - Motor de inferencia y entrenamiento basado en **TensorFlow/Keras**.
   - API RESTful mediante **Flask** para comunicación con el cliente.
   - Extrae 21 características (features) complejas por cada sorteo, incluyendo entropía de aparición, temperaturas ("hotness"), estacionalidad (fechas, turnos) y patrones repetitivos.

2. **Servidor y Frontend (Node.js)**
   - Servidor web ultraligero que entrega una interfaz de usuario interactiva y fluida.
   - Panel de control visual donde el usuario puede iniciar ciclos de entrenamiento, hacer scraping de datos históricos y solicitar predicciones al Oráculo.

---

## 🧠 Arquitectura de la Red Neuronal

El núcleo matemático reside en `hybrid_cnn_lstm.py`:
- **Conv1D Feature Extractor**: Analiza patrones locales a corto plazo.
- **Transformer Encoders**: Dos bloques de atención (Multi-Head Attention) para aprender relaciones complejas a largo plazo (tamaño de ventana = 30 sorteos).
- **Monte Carlo Dropout**: Aplicado durante la inferencia para calcular la "adivinanza empírica" o nivel de incertidumbre (desviación estándar de múltiples pasadas).

---

## 🚀 Instalación y Configuración

### Prerrequisitos
Asegúrate de tener instalado en tu sistema:
- **Python 3.8 o superior**
- **Node.js 18 o superior**

### 1. Clonar el Repositorio
```bash
git clone https://github.com/TU_USUARIO/AI_Oracle_Pick3.git
cd AI_Oracle_Pick3
```

### 2. Configurar Entorno de Python
Se recomienda el uso de entornos virtuales para evitar conflictos:
```bash
python -m venv venv
# Activar en Windows:
venv\Scripts\activate
# Activar en Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Entorno de Node.js
Instala cualquier dependencia subyacente (aunque el servidor web utiliza módulos nativos):
```bash
npm install
```

---

## 🖥️ Uso y Ejecución

La forma más sencilla de levantar todo el ecosistema (API + Web) en Windows es ejecutando el script proporcionado:

```cmd
INICIAR_TITAN.bat
```
*(Este script levantará la API de Flask en el puerto 5001 y el cliente web en el puerto 3000 automáticamente).*

### Ejecución Manual (Modo Desarrollo)

**Terminal 1 (Motor de IA):**
```bash
# Iniciar la API predictiva en el puerto 5001
python api.py
```

**Terminal 2 (Servidor Web):**
```bash
# Iniciar el servidor web de la interfaz de usuario en el puerto 3000
npm start
```

Una vez que ambos servicios estén corriendo, accede a la interfaz gráfica desde tu navegador en:
🔗 **`http://localhost:3000`**

---

## 📂 Estructura de Directorios y Archivos

| Archivo / Carpeta | Descripción |
| :--- | :--- |
| `api.py` | API Flask que carga los pesos, expone endpoints de `/train` y `/predict`. |
| `hybrid_cnn_lstm.py`| Definición matemática del modelo (CNN + Transformer) y lógica de Features. |
| `scrape.py` | Script de automatización (BeautifulSoup) para ingestar datos históricos. |
| `server.js` | Servidor Node.js que orquesta el frontend estático y se comunica con la API. |
| `index.html` | Interfaz gráfica principal del panel de control del Oráculo. |
| `/css` & `/js` | Assets estáticos para el frontend (Estilos y lógica cliente). |
| `INICIAR_TITAN.bat` | Script de inicialización rápida (1-click run) para Windows. |
| `cnn_lstm_weights.weights.h5` | Pesos entrenados de la red neuronal (generado tras entrenar). |

---

## 📡 Documentación de la API Interna

### `POST /api/python/train`
Entrena el modelo con el histórico de sorteos proporcionado.
- **Payload**: JSON con array `draws` (Mínimo 30 sorteos).
- **Acción**: Ejecuta `model.fit()` de manera asíncrona, guarda los pesos en `.h5`.

### `POST /api/python/predict`
Calcula el pronóstico para el siguiente sorteo basándose en la última ventana histórica.
- **Payload**: JSON con array `draws`.
- **Retorna**: Probabilidades de 0 a 9 para cada dígito (D1, D2, D3), más índices de entropía e incertidumbre MC.

---

## 📜 Licencia y Advertencia
*Este proyecto tiene fines estrictamente experimentales, educativos y de investigación sobre aprendizaje automático e inteligencia artificial. Los sistemas de lotería son juegos de azar puros; los autores de este código no garantizan ganancias y no se hacen responsables de pérdidas financieras.*
