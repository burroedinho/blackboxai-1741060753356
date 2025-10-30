"""
Configurações do Sistema Autônomo de Trading B3
"""
import os
from pathlib import Path

# ==================== CONFIGURAÇÕES GERAIS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STRATEGIES_DIR = BASE_DIR / "strategies"

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, STRATEGIES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== CONFIGURAÇÕES DE HARDWARE ====================
# Otimizado para ACER Predator Helios 300
GPU_MEMORY_LIMIT = 5120  # 5GB da RTX 3060 (deixa 1GB para sistema)
USE_GPU = True
MIXED_PRECISION = True  # FP16 para melhor performance
NUM_THREADS = 8  # i7 11th gen tem 8 threads

# ==================== CONFIGURAÇÕES DE TRADING ====================
# Ativos da B3 para análise
ATIVOS_B3 = [
    "PETR4.SA",  # Petrobras
    "VALE3.SA",  # Vale
    "ITUB4.SA",  # Itaú
    "BBDC4.SA",  # Bradesco
    "ABEV3.SA",  # Ambev
    "WEGE3.SA",  # WEG
    "RENT3.SA",  # Localiza
    "MGLU3.SA",  # Magazine Luiza
    "B3SA3.SA",  # B3
    "SUZB3.SA",  # Suzano
]

# Índices
INDICES = ["^BVSP"]  # Ibovespa

# Configurações de risco
CAPITAL_INICIAL = 10000.0  # R$ 10.000
RISCO_POR_OPERACAO = 0.02  # 2% por operação
MAX_DRAWDOWN = 0.15  # 15% de drawdown máximo
STOP_LOSS = 0.03  # 3% stop loss
TAKE_PROFIT = 0.06  # 6% take profit

# ==================== CONFIGURAÇÕES DE IA ====================
# Parâmetros do modelo
SEQUENCE_LENGTH = 60  # 60 períodos de histórico
FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12'
]
NUM_FEATURES = len(FEATURES)

# Treinamento
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ==================== AUTO-EVOLUÇÃO ====================
# Algoritmo Genético
POPULATION_SIZE = 20  # População de estratégias
GENERATIONS = 50  # Gerações de evolução
MUTATION_RATE = 0.15  # Taxa de mutação
CROSSOVER_RATE = 0.7  # Taxa de crossover
ELITE_SIZE = 3  # Melhores estratégias preservadas

# Critérios de fitness
FITNESS_WEIGHTS = {
    'sharpe_ratio': 0.3,
    'total_return': 0.25,
    'win_rate': 0.2,
    'max_drawdown': 0.15,
    'profit_factor': 0.1
}

# ==================== BACKTESTING ====================
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_END_DATE = "2024-12-31"
TIMEFRAME = "1d"  # 1 dia (pode ser: 1m, 5m, 15m, 1h, 1d)

# ==================== APIs GRATUITAS ====================
# Alpha Vantage (gratuito - 5 calls/min, 500 calls/dia)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")

# Yahoo Finance (gratuito - sem limite)
USE_YAHOO_FINANCE = True

# MetaTrader5 (gratuito - requer instalação)
USE_MT5 = False
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# ==================== DASHBOARD ====================
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
ENABLE_DASHBOARD = True

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "trading_system.log"

# ==================== MODO DE OPERAÇÃO ====================
PAPER_TRADING = True  # True = simulação, False = real
AUTO_TRADE = False  # True = executa ordens automaticamente
REWRITE_INTERVAL = 24  # Horas entre auto-reescritas do código

# ==================== SEGURANÇA ====================
ENABLE_ENCRYPTION = True
MAX_POSITION_SIZE = 0.1  # Máximo 10% do capital por posição
REQUIRE_CONFIRMATION = True  # Requer confirmação para trades reais
