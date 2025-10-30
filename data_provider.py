"""
Provedor de Dados da B3
Integra múltiplas APIs gratuitas para dados de mercado
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
import config
import ta  # Technical Analysis library

class B3DataProvider:
    """
    Provedor de dados da B3 usando APIs gratuitas
    """
    
    def __init__(self):
        self.ativos = config.ATIVOS_B3
        self.indices = config.INDICES
        logger.info(f"📊 Data Provider inicializado com {len(self.ativos)} ativos")
    
    def get_historical_data(self, ticker: str, period: str = "1y", 
                           interval: str = "1d") -> pd.DataFrame:
        """
        Busca dados históricos de um ativo
        
        Args:
            ticker: Código do ativo (ex: PETR4.SA)
            period: Período (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Intervalo (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame com OHLCV
        """
        try:
            logger.info(f"📥 Baixando dados de {ticker} - Período: {period}")
            
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"⚠️ Nenhum dado encontrado para {ticker}")
                return pd.DataFrame()
            
            # Renomeia colunas para lowercase
            df.columns = [col.lower() for col in df.columns]
            
            logger.success(f"✅ {len(df)} registros baixados para {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar dados de {ticker}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, ticker: str) -> Dict:
        """
        Busca dados em tempo real de um ativo
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            realtime_data = {
                'ticker': ticker,
                'price': info.get('currentPrice', 0),
                'open': info.get('open', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'timestamp': datetime.now()
            }
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados em tempo real de {ticker}: {e}")
            return {}
    
    def get_multiple_tickers(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Busca dados de múltiplos ativos
        """
        logger.info(f"📥 Baixando dados de {len(tickers)} ativos...")
        
        data = {}
        for ticker in tickers:
            df = self.get_historical_data(ticker, period)
            if not df.empty:
                data[ticker] = df
        
        logger.success(f"✅ Dados de {len(data)} ativos baixados")
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame
        """
        if df.empty:
            return df
        
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # ATR (Average True Range)
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX (Average Directional Index)
            df['adx'] = ta.trend.ADXIndicator(
                df['high'], df['low'], df['close']
            ).adx()
            
            # OBV (On-Balance Volume)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
            
            # Remove NaN
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            logger.success(f"✅ Indicadores técnicos adicionados")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar indicadores: {e}")
            return df
    
    def prepare_ml_data(self, df: pd.DataFrame, sequence_length: int = 60) -> tuple:
        """
        Prepara dados para machine learning
        
        Returns:
            X (features), y (labels)
        """
        if df.empty:
            return np.array([]), np.array([])
        
        # Adiciona indicadores
        df = self.add_technical_indicators(df)
        
        # Seleciona features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12'
        ]
        
        # Normaliza dados
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        df_scaled = df[feature_columns].copy()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_scaled),
            columns=feature_columns,
            index=df.index
        )
        
        # Cria labels (1 = sobe, 0 = desce, 2 = lateral)
        df['future_return'] = df['close'].pct_change(periods=1).shift(-1)
        df['label'] = 2  # Lateral por padrão
        df.loc[df['future_return'] > 0.01, 'label'] = 1  # Sobe
        df.loc[df['future_return'] < -0.01, 'label'] = 0  # Desce
        
        # Cria sequências
        X, y = [], []
        
        for i in range(sequence_length, len(df_scaled) - 1):
            X.append(df_scaled.iloc[i-sequence_length:i].values)
            y.append(df['label'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # One-hot encoding para labels
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(y, num_classes=3)
        
        logger.info(f"📊 Dados preparados: X shape={X.shape}, y shape={y.shape}")
        
        return X, y, scaler
    
    def get_market_sentiment(self) -> Dict:
        """
        Analisa sentimento geral do mercado
        """
        try:
            # Busca dados do Ibovespa
            ibov = self.get_historical_data("^BVSP", period="5d", interval="1d")
            
            if ibov.empty:
                return {'sentiment': 'neutral', 'score': 0}
            
            # Calcula variação
            last_close = ibov['close'].iloc[-1]
            prev_close = ibov['close'].iloc[-2]
            variation = (last_close - prev_close) / prev_close
            
            # Determina sentimento
            if variation > 0.02:
                sentiment = 'bullish'
                score = min(variation * 50, 1.0)
            elif variation < -0.02:
                sentiment = 'bearish'
                score = max(variation * 50, -1.0)
            else:
                sentiment = 'neutral'
                score = 0
            
            return {
                'sentiment': sentiment,
                'score': score,
                'ibov_price': last_close,
                'variation': variation
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao analisar sentimento: {e}")
            return {'sentiment': 'neutral', 'score': 0}


class AlphaVantageProvider:
    """
    Provider alternativo usando Alpha Vantage (API gratuita)
    Limite: 5 chamadas por minuto, 500 por dia
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"
        logger.info("🔑 Alpha Vantage Provider inicializado")
    
    def get_intraday_data(self, symbol: str, interval: str = "5min") -> pd.DataFrame:
        """
        Busca dados intraday
        """
        import requests
        
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series' not in str(data):
                logger.warning(f"⚠️ Dados não disponíveis para {symbol}")
                return pd.DataFrame()
            
            # Processa dados
            time_series_key = f'Time Series ({interval})'
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            logger.success(f"✅ Dados intraday de {symbol} obtidos")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados Alpha Vantage: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Teste do data provider
    logger.info("🧪 Testando Data Provider...")
    
    provider = B3DataProvider()
    
    # Testa download de dados
    df = provider.get_historical_data("PETR4.SA", period="1mo")
    logger.info(f"Dados PETR4: {len(df)} registros")
    
    # Testa indicadores técnicos
    df_with_indicators = provider.add_technical_indicators(df)
    logger.info(f"Colunas com indicadores: {df_with_indicators.columns.tolist()}")
    
    # Testa preparação para ML
    X, y, scaler = provider.prepare_ml_data(df)
    logger.info(f"Dados ML: X={X.shape}, y={y.shape}")
    
    # Testa sentimento
    sentiment = provider.get_market_sentiment()
    logger.info(f"Sentimento do mercado: {sentiment}")
    
    logger.success("✅ Testes concluídos!")
