import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
import numpy as np
import logging

# Configuração de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limitar o uso de memória da GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} GPUs físicas, {len(logical_gpus)} GPUs lógicas")
    except RuntimeError as e:
        print(e)

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModeloHibridoProfundo:
    def __init__(self, input_shape=(15, 6)):  # 15 velas, 6 features (OHLCV + momentum)
        logger.info(f"Inicializando Modelo Híbrido Profundo com input_shape={input_shape}")
        self.model = self._construir_modelo(input_shape)
        
    def _construir_modelo(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Camada Quântica Simulada
        x = Lambda(lambda x: tf.math.sin(x) * tf.math.cos(x))(inputs)
        
        # Rede LSTM Profunda
        lstm_out = LSTM(128, return_sequences=False)(x)
        
        # Mecanismo de Atenção Multi-Cabeça
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = LayerNormalization()(attention + inputs)
        attention = Dropout(0.3)(attention)
        attention = Lambda(lambda x: tf.reduce_mean(x, axis=1))(attention)  # Redução de dimensionalidade
        
        # Fusão de Recursos
        combined = tf.keras.layers.concatenate([lstm_out, attention])
        
        # Camadas Densas
        x = Dense(256, activation='swish')(combined)
        x = Dense(128, activation='swish')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
        return model

if __name__ == "__main__":
    # Teste básico do modelo
    modelo = ModeloHibridoProfundo()
    logger.info("Modelo criado com sucesso!")
