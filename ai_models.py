"""
Modelos de IA Multi-Arquitetura
Otimizado para NVIDIA RTX 3060 (6GB VRAM)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import config

# Configuração de GPU
def setup_gpu():
    """Configura GPU para uso otimizado"""
    # TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limita memória
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=config.GPU_MEMORY_LIMIT)]
            )
            
            # Mixed Precision para RTX 3060
            if config.MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            
            logger.info(f"✅ GPU configurada: {len(gpus)} GPU(s) disponível(is)")
            logger.info(f"🚀 Mixed Precision: {config.MIXED_PRECISION}")
            
        except RuntimeError as e:
            logger.error(f"Erro ao configurar GPU: {e}")
    else:
        logger.warning("⚠️ Nenhuma GPU detectada, usando CPU")
    
    # PyTorch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info(f"✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")

setup_gpu()


class TransformerModel(Model):
    """
    Modelo Transformer para séries temporais
    Baseado em "Attention is All You Need"
    """
    
    def __init__(self, seq_length=60, num_features=11, d_model=128, num_heads=8, 
                 num_layers=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding de entrada
        self.input_projection = layers.Dense(d_model)
        
        # Positional Encoding
        self.pos_encoding = self._positional_encoding(seq_length, d_model)
        
        # Transformer Encoder Layers
        self.encoder_layers = [
            self._transformer_encoder_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ]
        
        # Camadas de saída
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(3, activation='softmax')  # BUY, SELL, HOLD
    
    def _positional_encoding(self, seq_length, d_model):
        """Cria positional encoding"""
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def _transformer_encoder_layer(self, d_model, num_heads, dropout):
        """Cria uma camada de encoder"""
        return {
            'attention': layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
            'norm1': layers.LayerNormalization(),
            'ffn': keras.Sequential([
                layers.Dense(d_model * 4, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model)
            ]),
            'norm2': layers.LayerNormalization(),
            'dropout': layers.Dropout(dropout)
        }
    
    def call(self, inputs, training=False):
        # Projeção de entrada
        x = self.input_projection(inputs)
        
        # Adiciona positional encoding
        x = x + self.pos_encoding
        
        # Passa pelos encoders
        for encoder in self.encoder_layers:
            # Multi-head attention
            attn_output = encoder['attention'](x, x, training=training)
            attn_output = encoder['dropout'](attn_output, training=training)
            x = encoder['norm1'](x + attn_output)
            
            # Feed-forward
            ffn_output = encoder['ffn'](x, training=training)
            x = encoder['norm2'](x + ffn_output)
        
        # Pooling e saída
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class LSTMModel(Model):
    """
    Modelo LSTM Bidirecional com Attention
    """
    
    def __init__(self, seq_length=60, num_features=11):
        super().__init__()
        
        # LSTM Layers
        self.lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        
        # Attention Layer
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.layer_norm = layers.LayerNormalization()
        
        # Dense Layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(0.3)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(3, activation='softmax')
    
    def call(self, inputs, training=False):
        # LSTM
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        
        # Attention
        attn_output = self.attention(x, x, training=training)
        x = self.layer_norm(x + attn_output)
        
        # Output
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class CNNModel(Model):
    """
    Modelo CNN 1D para padrões de preço
    """
    
    def __init__(self, seq_length=60, num_features=11):
        super().__init__()
        
        # Convolutional Layers
        self.conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2)
        
        self.conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2)
        
        self.conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')
        self.pool3 = layers.GlobalMaxPooling1D()
        
        # Dense Layers
        self.dropout = layers.Dropout(0.3)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(3, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class EnsembleModel:
    """
    Ensemble de múltiplos modelos para melhor acurácia
    """
    
    def __init__(self, seq_length=60, num_features=11):
        logger.info("🤖 Inicializando Ensemble de Modelos")
        
        self.transformer = TransformerModel(seq_length, num_features)
        self.lstm = LSTMModel(seq_length, num_features)
        self.cnn = CNNModel(seq_length, num_features)
        
        # Pesos do ensemble (podem ser otimizados)
        self.weights = {
            'transformer': 0.4,
            'lstm': 0.35,
            'cnn': 0.25
        }
        
        # Compila modelos
        optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        loss = keras.losses.CategoricalCrossentropy()
        metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        
        for model in [self.transformer, self.lstm, self.cnn]:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.success("✅ Ensemble criado com sucesso")
    
    def predict(self, X):
        """Predição usando ensemble"""
        # Predições individuais
        pred_transformer = self.transformer.predict(X, verbose=0)
        pred_lstm = self.lstm.predict(X, verbose=0)
        pred_cnn = self.cnn.predict(X, verbose=0)
        
        # Weighted average
        ensemble_pred = (
            pred_transformer * self.weights['transformer'] +
            pred_lstm * self.weights['lstm'] +
            pred_cnn * self.weights['cnn']
        )
        
        return ensemble_pred
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Treina todos os modelos"""
        logger.info("🎯 Iniciando treinamento do Ensemble")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                str(config.MODELS_DIR / 'best_model_{epoch:02d}.h5'),
                save_best_only=True
            )
        ]
        
        histories = {}
        
        # Treina Transformer
        logger.info("📊 Treinando Transformer...")
        histories['transformer'] = self.transformer.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Treina LSTM
        logger.info("📊 Treinando LSTM...")
        histories['lstm'] = self.lstm.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Treina CNN
        logger.info("📊 Treinando CNN...")
        histories['cnn'] = self.cnn.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.success("✅ Treinamento concluído!")
        return histories
    
    def save_models(self, prefix="ensemble"):
        """Salva todos os modelos"""
        self.transformer.save(config.MODELS_DIR / f"{prefix}_transformer.h5")
        self.lstm.save(config.MODELS_DIR / f"{prefix}_lstm.h5")
        self.cnn.save(config.MODELS_DIR / f"{prefix}_cnn.h5")
        logger.info(f"💾 Modelos salvos em {config.MODELS_DIR}")
    
    def load_models(self, prefix="ensemble"):
        """Carrega modelos salvos"""
        try:
            self.transformer = keras.models.load_model(
                config.MODELS_DIR / f"{prefix}_transformer.h5"
            )
            self.lstm = keras.models.load_model(
                config.MODELS_DIR / f"{prefix}_lstm.h5"
            )
            self.cnn = keras.models.load_model(
                config.MODELS_DIR / f"{prefix}_cnn.h5"
            )
            logger.success("✅ Modelos carregados com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")


class ReinforcementLearningAgent:
    """
    Agente de Reinforcement Learning usando PPO
    Para decisões de trading em tempo real
    """
    
    def __init__(self):
        logger.info("🎮 Inicializando Agente RL (PPO)")
        # Implementação com Stable-Baselines3 será adicionada
        pass
    
    def train(self, env, total_timesteps=100000):
        """Treina agente no ambiente"""
        pass
    
    def predict(self, observation):
        """Prediz ação"""
        pass


if __name__ == "__main__":
    # Teste dos modelos
    logger.info("🧪 Testando modelos...")
    
    # Dados dummy
    X_dummy = np.random.randn(100, 60, 11)
    y_dummy = np.random.randint(0, 3, (100, 3))
    
    # Cria ensemble
    ensemble = EnsembleModel()
    
    # Teste de predição
    predictions = ensemble.predict(X_dummy[:10])
    logger.info(f"Predições shape: {predictions.shape}")
    logger.success("✅ Teste concluído!")
