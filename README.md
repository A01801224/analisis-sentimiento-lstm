# Análisis de sentimiento en tweets de aerolíneas con LSTM

**Autor:** Emilio Páez de la Mora — A01801224
**Módulo 5:** Implementación de un modelo de deep learning

Modelo de deep learning (Keras + TensorFlow) que clasifica el sentimiento de tweets dirigidos
a aerolíneas de EE. UU. como **negativo, neutral o positivo**, usando una red recurrente LSTM.

## Contenido del repositorio
| Archivo | Descripción |
|---|---|
| `analisis_sentimiento_lstm.ipynb` | Reporte completo: introducción, datos, desarrollo y ajuste del modelo, resultados, conclusiones y aplicación |
| `modelo_final.keras` | Modelo final entrenado (arquitectura + pesos) |
| `vocabulario.txt` | Vocabulario de 5,000 palabras usado para convertir el texto a números (necesario para la aplicación) |

## Datos
**Twitter US Airline Sentiment** (14,640 tweets, febrero 2015), originalmente de CrowdFlower.
Fuente: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

El dataset no se incluye en el repositorio: el notebook lo descarga automáticamente con `kagglehub`.

## Modelo final
`Embedding(5000, 32)` → `Dropout(0.3)` → `LSTM(64)` → `Dropout(0.5)` → `Dense(3, softmax)`,
entrenado con EarlyStopping.

| Versión | Cambio | F1 macro (validación) |
|---|---|---|
| Base | Sin regularización | 0.680 |
| **V1 (final)** | **Embedding 32 + Dropout + EarlyStopping** | **0.740** |
| V2 | V1 + pesos por clase + bidireccional | 0.703 |
| V3 | V1 + bidireccional | 0.741 |

**Resultado en el conjunto de prueba:** accuracy **78.6%** y F1 macro **0.727**
(línea base: 62.9%).

## Cómo usarlo
1. Abrir el notebook en Google Colab (con GPU: *Entorno de ejecución → Cambiar tipo → T4 GPU*).
2. Ejecutar las celdas en orden. La sección **7. Aplicación** contiene la función
   `predecir_sentimiento(texto)`, que recibe un tweet y regresa su sentimiento:

```python
predecir_sentimiento("@united my flight was delayed 4 hours and nobody told us anything")
# Sentimiento: NEGATIVE (99.5%)
```

Para usar solo la aplicación sin reentrenar, basta con subir `modelo_final.keras` y
`vocabulario.txt` a Colab y ejecutar las celdas de importación, `limpiar_texto` y la Sección 7.

## Tecnologías
Python · TensorFlow / Keras · pandas · scikit-learn · matplotlib · seaborn