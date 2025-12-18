import asyncio
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError, BadRequest
import json
from pathlib import Path
import os
import pytz

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

TG_TOKEN = os.getenv("TG_TOKEN", "")

PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "")
PROXY_USER = os.getenv("PROXY_USER", "")
PROXY_PASS = os.getenv("PROXY_PASS", "")

if PROXY_HOST and PROXY_PORT:
    if PROXY_USER and PROXY_PASS:
        PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
    else:
        PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
else:
    PROXY_URL = None


class CryptoPumpPredictorBot:
    def __init__(
        self,
        telegram_token: str,
        model_path: str | None = None,
        proxy_url: str | None = None,
    ):
        self.telegram_token = telegram_token
        self.proxy_url = proxy_url

        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "pump_predictor_model.pkl")

        builder = Application.builder().token(telegram_token)

        if self.proxy_url:
            builder = (
                builder
                .proxy(self.proxy_url)
                .get_updates_pool_timeout(30)
                .read_timeout(20)
                .write_timeout(20)
                .connect_timeout(10)
            )
        else:
            builder = (
                builder
                .get_updates_pool_timeout(30)
                .read_timeout(20)
                .write_timeout(20)
                .connect_timeout(10)
            )

        self.app = builder.build()

        self.tz_msk = pytz.timezone("Europe/Moscow")

        self.last_heartbeat_msg: dict[int, int] = {}

        self.subscribers_file = "subscribers.json"
        self.subscribers = self.load_subscribers()

        logger.info(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        logger.info("✅ Model loaded")
        logger.info(f"MODEL FEATURES: {self.feature_names}")

        self.exchanges = self.init_exchanges()

        self.top_symbols_per_exchange = 200
        self.lookback_candles = 20
        self.scan_interval = int(os.getenv("SCAN_INTERVAL", "60"))
        self.ml_threshold = float(os.getenv("ML_THRESHOLD", "0.35"))

        self.sent_signals: set[str] = set()
        self.symbols_cache: dict[str, list[str]] = {}
        self.symbols_cache_time: dict[str, float] = {}
        self.symbols_cache_ttl = 600

        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("help", self.cmd_help))

    def load_subscribers(self) -> set[int]:
        if Path(self.subscribers_file).exists():
            with open(self.subscribers_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_subscribers(self) -> None:
        with open(self.subscribers_file, "w") as f:
            json.dump(list(self.subscribers), f)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        if chat_id not in self.subscribers:
            self.subscribers.add(chat_id)
            self.save_subscribers()

            await update.message.reply_text(
                "✅ <b>Subscription activated!</b>\n\n"
                "You will receive real-time pump signals based on the ML model.\n\n"
                "<b>Commands:</b>\n"
                "/stop – unsubscribe\n"
                "/status – bot status\n"
                "/help – help and description",
                parse_mode="HTML",
            )
            logger.info(f"➕ New subscriber: {chat_id}")
        else:
            await update.message.reply_text(
                "ℹ️ You are already subscribed.",
                parse_mode="HTML",
            )

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        if chat_id in self.subscribers:
            self.subscribers.remove(chat_id)
            self.save_subscribers()
            self.last_heartbeat_msg.pop(chat_id, None)

            await update.message.reply_text(
                "❌ <b>Subscription cancelled</b>\n\n"
                "You will no longer receive signals.\n"
                "Use /start to subscribe again.",
                parse_mode="HTML",
            )
            logger.info(f"➖ Unsubscribed: {chat_id}")
        else:
            await update.message.reply_text("ℹ️ You are not subscribed.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        total_subscribers = len(self.subscribers)
        total_exchanges = len(self.exchanges)

        status_msg = (
            "📊 <b>Bot status</b>\n\n"
            f"👥 Subscribers: <b>{total_subscribers}</b>\n"
            f"🏦 Exchanges: <b>{total_exchanges}</b>\n"
            f"🤖 ML threshold: <b>{self.ml_threshold * 100:.0f}%</b>\n"
            f"⏱ Scan interval: <b>{self.scan_interval}s</b>\n\n"
            "Exchanges:\n"
        )

        for name in self.exchanges.keys():
            status_msg += f"  • {name.title()}\n"

        await update.message.reply_text(status_msg, parse_mode="HTML")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_msg = (
            "<b>🤖 Crypto Pump Predictor Bot</b>\n\n"
            "<b>Commands:</b>\n"
            "/start – subscribe to signals\n"
            "/stop – unsubscribe\n"
            "/status – bot status\n"
            "/help – this help\n\n"
            "<b>What does the bot do?</b>\n"
            "It scans top symbols on Binance, Bybit and MEXC and detects pump-like patterns "
            "using a machine learning model.\n\n"
            "<b>Signal strength:</b>\n"
            "🔥 STRONG – probability ≥ 50%\n"
            "⚡ MEDIUM – probability 40–50%\n"
            "💡 WEAK – probability 35–40%\n\n"
            "<b>Exchanges:</b>\n"
            "🟡 Binance\n"
            "🟠 Bybit\n"
            "🟢 MEXC\n\n"
            "⚠️ This bot is for educational and research purposes only.\n"
            "Trading involves risk. Use at your own responsibility."
        )

        await update.message.reply_text(help_msg, parse_mode="HTML")

    async def heartbeat_job(self, context: ContextTypes.DEFAULT_TYPE):
        if not self.subscribers:
            return

        now_msk = datetime.now(self.tz_msk).strftime("%Y-%m-%d %H:%M:%S MSK")
        msg = (
            "🫀 <b>Heartbeat</b>\n"
            f"Bot is running. Time: <b>{now_msk}</b>\n"
            f"Subscribers: <b>{len(self.subscribers)}</b>\n"
            f"Exchanges: <b>{len(self.exchanges)}</b>\n"
            f"Scan interval: <b>{self.scan_interval}s</b>"
        )

        for chat_id in list(self.subscribers):
            try:
                old_id = self.last_heartbeat_msg.get(chat_id)
                if old_id:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=old_id)
                    except BadRequest as e:
                        logger.info(f"Heartbeat delete skipped for {chat_id}: {e}")
                    except TelegramError as e:
                        logger.info(f"Heartbeat delete error for {chat_id}: {e}")
                    except Exception as e:
                        logger.info(f"Heartbeat delete unexpected for {chat_id}: {e}")

                m = await context.bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

                self.last_heartbeat_msg[chat_id] = m.message_id

            except TelegramError as e:
                if "bot was blocked" in str(e).lower():
                    self.subscribers.remove(chat_id)
                    self.save_subscribers()
                    self.last_heartbeat_msg.pop(chat_id, None)
                else:
                    logger.error(f"Heartbeat send error for {chat_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected heartbeat error for {chat_id}: {e}")

        logger.info(f"🫀 Heartbeat sent ({now_msk})")

    def schedule_heartbeat(self) -> None:
        jq = self.app.job_queue
        if jq is None:
            logger.warning("JobQueue is not available. Install python-telegram-bot[job-queue].")
            return

        from datetime import time as dtime

        for h in range(24):
            jq.run_daily(
                callback=self.heartbeat_job,
                time=dtime(hour=h, minute=0, second=0, tzinfo=self.tz_msk),
                days=(0, 1, 2, 3, 4, 5, 6),
                name=f"heartbeat_{h:02d}msk",
            )

        logger.info("✅ Heartbeat scheduled hourly at every full hour (MSK).")

    def init_exchanges(self) -> dict:
        exchanges: dict[str, ccxt.Exchange] = {}

        proxy_cfg: dict[str, dict] = {}
        if self.proxy_url:
            proxy_cfg = {
                "proxies": {
                    "http": self.proxy_url,
                    "https": self.proxy_url,
                }
            }

        try:
            exchanges["binance"] = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    **proxy_cfg,
                }
            )
            exchanges["binance"].options["defaultType"] = "future"
            logger.info("✅ Binance connected")
        except Exception as e:
            logger.error(f"❌ Binance: {e}")

        try:
            exchanges["bybit"] = ccxt.bybit(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    **proxy_cfg,
                }
            )
            exchanges["bybit"].options["defaultType"] = "swap"
            logger.info("✅ Bybit connected")
        except Exception as e:
            logger.error(f"❌ Bybit: {e}")

        try:
            exchanges["mexc"] = ccxt.mexc(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    **proxy_cfg,
                }
            )
            exchanges["mexc"].options["defaultType"] = "swap"
            logger.info("✅ MEXC connected")
        except Exception as e:
            logger.error(f"❌ MEXC: {e}")

        logger.info(f"Active exchanges: {len(exchanges)}")
        return exchanges

    def get_top_symbols(self, exchange_name: str, exchange: ccxt.Exchange) -> list[str]:
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                markets = exchange.load_markets()
                tickers = exchange.fetch_tickers()

                usdt_futures: list[tuple[str, float]] = []
                for symbol, market in markets.items():
                    market_type = market.get("type", "")
                    is_futures = (
                        market.get("swap")
                        or market.get("future")
                        or market.get("linear")
                        or market_type in ["swap", "future", "linear"]
                    )

                    if not is_futures:
                        continue

                    quote = market.get("quote", "")
                    if quote not in ["USDT", "USD"]:
                        continue

                    if not market.get("active", True):
                        continue

                    ticker = tickers.get(symbol, {})
                    volume = ticker.get("quoteVolume", 0) or 0

                    if volume > 0:
                        usdt_futures.append((symbol, volume))

                usdt_futures.sort(key=lambda x: x[1], reverse=True)
                top_symbols = [s for s, _ in usdt_futures[: self.top_symbols_per_exchange]]

                stablecoins = ["USDE", "BFUSD", "XUSD", "USDT", "USDC", "BUSD", "DAI", "FDUSD"]
                top_symbols = [s for s in top_symbols if s.split("/")[0] not in stablecoins]

                logger.info(f"✓ {exchange_name}: {len(top_symbols)} symbols")
                return top_symbols

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠ {exchange_name} attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"✗ {exchange_name} FAILED — {e}")
                    return []

        return []

    def fetch_recent_candles(self, exchange: ccxt.Exchange, symbol: str) -> pd.DataFrame | None:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, "5m", limit=30)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["volume_usdt"] = df["volume"] * df["close"]
                return df
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                continue
        return None

    def calculate_features(self, df: pd.DataFrame) -> dict | None:
        if len(df) < self.lookback_candles:
            return None

        latest = df.iloc[-1]
        price_change = (latest["close"] - latest["open"]) / latest["open"] * 100
        vol_mean_20 = (
            df["volume_usdt"].iloc[:-1].tail(self.lookback_candles).mean()
        )
        vol_ratio = latest["volume_usdt"] / vol_mean_20 if vol_mean_20 > 0 else 0

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        volatility = (
            df["close"].rolling(20).std().iloc[-1]
            / df["close"].rolling(20).mean().iloc[-1]
        )

        if len(df) >= 6:
            trend_5 = (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100
        else:
            trend_5 = 0

        return {
            "vol_ratio": vol_ratio,
            "price_change": price_change,
            "volume_usdt": latest["volume_usdt"],
            "rsi": rsi,
            "volatility": volatility,
            "trend_5": trend_5,
        }

    def predict_pump(self, features: dict) -> float:
        feature_vector = [features[name] for name in self.feature_names]
        proba = self.model.predict_proba([feature_vector])[0, 1]
        return float(proba)

    def scan_exchange(
        self,
        exchange_name: str,
        exchange: ccxt.Exchange,
        symbols: list[str],
    ) -> list[dict]:
        signals: list[dict] = []

        for symbol in symbols:
            try:
                df = self.fetch_recent_candles(exchange, symbol)
                if df is None or len(df) < self.lookback_candles:
                    continue

                features = self.calculate_features(df)
                if features is None:
                    continue

                if features["volume_usdt"] < 200_000:
                    continue
                if features["vol_ratio"] < 2:
                    continue
                if abs(features["price_change"]) > 15:
                    continue

                pump_probability = self.predict_pump(features)
                logger.info(
                    f"[DBG] {exchange_name} {symbol} "
                    f"proba={pump_probability:.3f} "
                    f"vol_usdt={features['volume_usdt']:.0f} "
                    f"vr={features['vol_ratio']:.2f} "
                    f"pc={features['price_change']:.2f}"
                )

                if pump_probability >= self.ml_threshold:
                    signal_key = f"{exchange_name}:{symbol}"

                    if signal_key not in self.sent_signals:
                        signals.append(
                            {
                                "exchange": exchange_name,
                                "symbol": symbol,
                                "probability": pump_probability,
                                "vol_ratio": features["vol_ratio"],
                                "price_change": features["price_change"],
                                "volume_usdt": features["volume_usdt"],
                                "rsi": features["rsi"],
                                "timestamp": datetime.now(),
                            }
                        )
                        self.sent_signals.add(signal_key)

            except Exception as e:
                logger.warning(f"[ERR] {exchange_name} {symbol}: {e}")
                continue

        return signals

    def scan_all_exchanges(self) -> list[dict]:
        logger.info("🔍 Fetching top symbols...")

        current_time = time.time()
        exchange_symbols: dict[str, list[str]] = {}

        for name, exchange in self.exchanges.items():
            try:
                cache_age = current_time - self.symbols_cache_time.get(name, 0)

                if name in self.symbols_cache and cache_age < self.symbols_cache_ttl:
                    symbols = self.symbols_cache[name]
                    logger.info(f"📦 {name}: {len(symbols)} symbols (from cache)")
                    exchange_symbols[name] = symbols
                else:
                    symbols = self.get_top_symbols(name, exchange)
                    if symbols:
                        self.symbols_cache[name] = symbols
                        self.symbols_cache_time[name] = current_time
                        exchange_symbols[name] = symbols
                    else:
                        if name in self.symbols_cache:
                            logger.warning(f"⚠ {name}: using old cache")
                            exchange_symbols[name] = self.symbols_cache[name]
                        else:
                            exchange_symbols[name] = []
            except Exception as e:
                logger.error(f"✗ {name}: {e}")
                if name in self.symbols_cache:
                    exchange_symbols[name] = self.symbols_cache[name]
                else:
                    exchange_symbols[name] = []

        total_symbols = sum(len(s) for s in exchange_symbols.values())
        logger.info(f"📊 Total symbols: {total_symbols}")

        if total_symbols == 0:
            return []

        all_signals: list[dict] = []

        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures: dict = {}
            for name, exchange in self.exchanges.items():
                if name in exchange_symbols and exchange_symbols[name]:
                    futures[executor.submit(
                        self.scan_exchange,
                        name,
                        exchange,
                        exchange_symbols[name],
                    )] = name

            for future in as_completed(futures):
                exchange_name = futures[future]
                try:
                    signals = future.result()
                    all_signals.extend(signals)
                    logger.info(f"✓ {exchange_name}: {len(signals)} signals")
                except Exception as e:
                    logger.error(f"✗ {exchange_name}: {e}")

        return all_signals

    async def broadcast_signal(self, message: str) -> None:
        if not self.subscribers:
            logger.warning("⚠ No subscribers")
            return

        success_count = 0
        fail_count = 0

        for chat_id in list(self.subscribers):
            try:
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                success_count += 1

            except TelegramError as e:
                fail_count += 1
                if "bot was blocked" in str(e).lower():
                    logger.warning(
                        f"❌ User {chat_id} blocked the bot — removing subscriber"
                    )
                    self.subscribers.remove(chat_id)
                    self.save_subscribers()
                    self.last_heartbeat_msg.pop(chat_id, None)
                else:
                    logger.error(f"❌ Send error {chat_id}: {e}")
            except Exception as e:
                fail_count += 1
                logger.error(f"❌ Unexpected send error {chat_id}: {e}")

        logger.info(f"📤 Sent: {success_count} OK, {fail_count} errors")

    def build_links(self, exchange: str, symbol: str) -> tuple[str | None, str | None]:
        try:
            base_quote = symbol.split(":")[0]
            base, quote = base_quote.split("/")
        except ValueError:
            return None, None

        pair = f"{base}{quote}"
        exch_url = ""
        tv_url = ""

        ex_name = exchange.lower()

        if ex_name == "binance":
            exch_url = f"https://www.binance.com/en/futures/{pair}"
            tv_prefix = "BINANCE"
        elif ex_name == "bybit":
            exch_url = f"https://www.bybit.com/trade/usdt/{pair}"
            tv_prefix = "BYBIT"
        elif ex_name == "mexc":
            exch_url = f"https://futures.mexc.com/exchange/{base}_{quote}"
            tv_prefix = "MEXC"
        else:
            tv_prefix = exchange.upper()

        tv_url = f"https://www.tradingview.com/chart/?symbol={tv_prefix}:{pair}"

        return exch_url, tv_url

    def format_signal_message(self, signal: dict) -> str:
        emoji_map = {
            "binance": "🟡",
            "bybit": "🟠",
            "mexc": "🟢",
        }

        emoji = emoji_map.get(signal["exchange"], "⚪")

        prob = signal["probability"]
        if prob >= 0.5:
            strength = "🔥 STRONG"
        elif prob >= 0.4:
            strength = "⚡ MEDIUM"
        else:
            strength = "💡 WEAK"

        exchange_url, tv_url = self.build_links(signal["exchange"], signal["symbol"])

        links_block = ""
        if exchange_url:
            links_block += f'\n🔗 <a href="{exchange_url}">Open on exchange</a>'
        if tv_url:
            links_block += f'\n📈 <a href="{tv_url}">View on TradingView</a>'

        message = (
            f"{emoji} <b>{signal['exchange'].upper()}</b> | {strength}\n\n"
            f"<b>🪙 {signal['symbol']}</b>\n\n"
            f"🤖 ML Probability: <b>{signal['probability'] * 100:.1f}%</b>\n"
            f"📊 Volume: <b>${signal['volume_usdt'] / 1e6:.1f}M</b> "
            f"({signal['vol_ratio']:.1f}x avg)\n"
            f"📈 Price Change: <b>{signal['price_change']:.2f}%</b>\n"
            f"📉 RSI: <b>{signal['rsi']:.1f}</b>\n\n"
            f"⏰ {signal['timestamp'].strftime('%H:%M:%S')}"
            f"{links_block}"
        )

        return message

    def cleanup_old_signals(self) -> None:
        self.sent_signals.clear()

    async def scanning_loop(self) -> None:
        logger.info("🔄 Starting scanning loop...")

        scan_count = 0
        while True:
            try:
                scan_count += 1
                logger.info(
                    f"🔄 Scan #{scan_count} | {datetime.now().strftime('%H:%M:%S')}"
                )

                loop = asyncio.get_event_loop()
                signals = await loop.run_in_executor(None, self.scan_all_exchanges)

                if signals:
                    logger.info(f"🚨 Found {len(signals)} signals")
                    signals = sorted(
                        signals,
                        key=lambda x: x["probability"],
                        reverse=True,
                    )

                    for signal in signals:
                        message = self.format_signal_message(signal)
                        await self.broadcast_signal(message)
                        await asyncio.sleep(1)
                else:
                    logger.info("✅ No signals")

                if scan_count % 120 == 0:
                    self.cleanup_old_signals()
                    logger.info("🧹 Signal cache cleared")

                logger.info(
                    f"⏳ Next scan in {self.scan_interval}s..."
                )
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                logger.error(f"❌ Error in scanning loop: {e}")
                await asyncio.sleep(10)

    def run(self) -> None:
        if not self.telegram_token:
            logger.error("TG_TOKEN environment variable is not set. Exiting.")
            raise SystemExit(1)

        logger.info("=" * 80)
        logger.info("🤖 TELEGRAM CRYPTO PUMP PREDICTOR BOT")
        logger.info("=" * 80)
        logger.info(f"Subscribers: {len(self.subscribers)}")
        logger.info(f"Exchanges: {len(self.exchanges)}")
        logger.info(f"ML threshold: {self.ml_threshold * 100:.0f}%")
        logger.info("=" * 80)

        async def post_init(app: Application):
            self.schedule_heartbeat()
            asyncio.create_task(self.scanning_loop())

        self.app.post_init = post_init

        logger.info("🚀 Bot started.")
        logger.info("Use /start to subscribe to signals.")
        self.app.run_polling()


def main():
    bot = CryptoPumpPredictorBot(
        telegram_token=TG_TOKEN,
        model_path=os.getenv("MODEL_PATH", "pump_predictor_model.pkl"),
        proxy_url=PROXY_URL,
    )
    bot.run()


if __name__ == "__main__":
    main()
