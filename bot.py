import pandas as pd
import numpy as np
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class TradingBot:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def get_data(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def add_features(self, df):
        # Indicateurs techniques
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Retours sur différentes périodes
        df['return_1h'] = df['close'].pct_change()
        df['return_4h'] = df['close'].pct_change(4)
        df['return_24h'] = df['close'].pct_change(24)
        
        # Volatilité
        df['volatility'] = df['return_1h'].rolling(24).std()
        
        # Volume moyen
        df['volume_ma'] = df['volume'].rolling(24).mean()
        
        return df
    
    def prepare_data(self, df):
        df = self.add_features(df)
        
        # Création de la variable cible (1 si le prix monte dans l'heure suivante, 0 sinon)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # Sélection des features
        features = ['SMA_20', 'SMA_50', 'return_1h', 'return_4h', 'return_24h', 'volatility', 'volume_ma']
        
        # Suppression des lignes avec des valeurs manquantes
        df = df.dropna().reset_index(drop=True)  # Réinitialisation des indices
        
        X = df[features]
        y = df['target']
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df
    
    def train_model(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def analyze(self, symbol, timeframe):
        # Récupération des données
        df = self.get_data(symbol, timeframe)
        
        # Préparation des données
        X, y, df = self.prepare_data(df)
        
        # Division des données
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Entraînement du modèle
        self.train_model(X_train, y_train)
        
        # Prédictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calcul de la précision
        accuracy = (predictions == y_test).mean()
        
        return df, predictions, probabilities, accuracy
