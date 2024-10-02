import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from bot import TradingBot
import numpy as np

def create_chart(df, predictions, probabilities):
    # Graphique des prix
    fig = go.Figure(data=[go.Candlestick(x=df.index[-len(predictions):],
                                         open=df['open'][-len(predictions):],
                                         high=df['high'][-len(predictions):],
                                         low=df['low'][-len(predictions):],
                                         close=df['close'][-len(predictions):])])
    
    # Ajout des moyennes mobiles
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50"))
    
    # Ajout des signaux d'achat (1) et de vente (0)
    buy_signals = np.where(predictions == 1)[0]
    sell_signals = np.where(predictions == 0)[0]

    # Utiliser l'index des prédictions pour sélectionner les valeurs correctes dans le DataFrame
    prediction_index = df.index[-len(predictions):]

    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=prediction_index[buy_signals],
            y=df['high'][-len(predictions):].values[buy_signals] + (df['high'][-len(predictions):].values[buy_signals] * 0.002),
            mode='markers',
            name='Signaux d\'achat',
            marker=dict(symbol='triangle-up', size=10, color='green'),
        ))
    
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=prediction_index[sell_signals],
            y=df['low'][-len(predictions):].values[sell_signals] - (df['low'][-len(predictions):].values[sell_signals] * 0.002),
            mode='markers',
            name='Signaux de vente',
            marker=dict(symbol='triangle-down', size=10, color='red'),
        ))
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def run_app():
    st.set_page_config(page_title="Bot de Trading ML", layout="wide")
    bot = TradingBot()

    st.title("🤖 Bot de Trading Crypto avec ML")

    col1, col2 = st.columns([3, 1])

    with col2:
        symbol = st.selectbox("Choisir une crypto", ["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        timeframe = st.selectbox("Intervalle", ["1h", "4h", "1d"])

    if st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            df, predictions, probabilities, accuracy = bot.analyze(symbol, timeframe)
            
            # Affichage du graphique
            st.plotly_chart(create_chart(df, predictions, probabilities), use_container_width=True)
            
            # Métriques dans une grille de 3 colonnes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Précision du modèle", f"{accuracy*100:.2f}%")
            
            with col2:
                derniers_prix = df['close'].iloc[-1]
                st.metric("Dernier prix", f"{derniers_prix:.2f} USDT")
            
            with col3:
                signal = "ACHAT 🟢" if predictions[-1] == 1 else "VENTE 🔴"
                confiance = probabilities[-1].max()
                st.metric("Signal actuel", signal, f"Confiance: {confiance*100:.2f}%")
            
            # Statistiques des signaux
            st.subheader("Statistiques des signaux")
            col4, col5, col6 = st.columns(3)
            
            nb_buy = sum(predictions == 1)
            nb_sell = sum(predictions == 0)
            
            with col4:
                st.metric("Nombre de signaux d'achat", nb_buy)
            with col5:
                st.metric("Nombre de signaux de vente", nb_sell)
            with col6:
                ratio = nb_buy / (nb_buy + nb_sell) * 100 if (nb_buy + nb_sell) > 0 else 0
                st.metric("Ratio achat/vente", f"{ratio:.1f}%")
            
            # Affichage des features importantes
            st.subheader("Importance des caractéristiques")
            feature_imp = pd.DataFrame({
                'feature': ['SMA_20', 'SMA_50', 'return_1h', 'return_4h', 'return_24h', 'volatility', 'volume_ma'],
                'importance': bot.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.bar_chart(feature_imp.set_index('feature'))

if __name__ == "__main__":
    run_app()
