# Crypto Pump Predictor (Telegram Bot)

Telegram bot that scans Binance, Bybit and MEXC futures and uses a **machine learning model** to detect pump-like patterns.  
Signals are sent to all subscribed chats with probability, volume, price change and direct links to exchanges and TradingView.

## Features

- 🔍 Exchanges:
  - Binance (USDT futures)
  - Bybit (USDT perpetual)
  - MEXC (USDT perpetual)
- 🤖 ML-based pump detection:
  - Uses pre-trained model (`pump_predictor_model.pkl`)
  - Feature set loaded from the model file (`feature_names`)
- 📊 Per-symbol features:
  - 5m candles (last 30 bars)
  - Volume in USDT and volume ratio vs. average
  - Price change
  - RSI
  - Volatility
  - Short-term trend (last 5 candles)
- 📨 Subscriptions:
  - `/start` – subscribe
  - `/stop` – unsubscribe
  - Signals are sent only once per symbol and exchange (deduplicated)
- 🫀 Heartbeat:
  - Hourly heartbeat message to each chat
  - Previous heartbeat message is deleted and replaced with a new one
- ⚙️ Configurable:
  - Scan interval
  - ML probability threshold
  - Proxy settings
  - Model path


