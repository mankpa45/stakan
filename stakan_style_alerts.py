"""
Stakan-style crypto alert bot
=============================

Monitors ALL Binance USDT spot pairs every 30 seconds and sends a Telegram
alert on either of these signals:

  REVERSAL + VOLUME     - price was trending green (up) over the last
                   FLIP_LOOKBACK_MINUTES, then the most recent poll
                   interval turns red, AND volume over the last hour
                   is elevated at least FLIP_VOLUME_MULTIPLIER vs the
                   hour before it -> SELL alert.
                   The mirror case (red trend -> green flip, same
                   volume confirmation) -> BUY alert.

  CONTINUATION + VOLUME - price was already trending in one direction
                   over FLIP_LOOKBACK_MINUTES, and the most recent poll
                   interval keeps moving the SAME direction (no flip),
                   AND volume is elevated at least CONTINUATION_VOLUME_MULTIPLIER
                   -> CONTINUATION UP or CONTINUATION DOWN alert. This
                   uses a stricter volume bar than reversals, to keep
                   continuation alerts less frequent. Useful for trend-
                   following instead of catching reversals.

HOW VOLUME IS MEASURED
-----------------------
Earlier versions of this script estimated volume by diffing Binance's
rolling 24h ticker "volume" field between polls. That field is a rolling
trailing-24h window, not a running counter, so the diff is noisy and
frequently ~0 even during real trading activity - a design flaw, not a
"needs more time" issue.

This version fixes that with a two-stage approach:
  1. Every cycle, one cheap call (`/api/v3/ticker/24hr`, all symbols at
     once) tracks price only, and finds symbols with a real price trend
     or flip - this part is free and instant.
  2. Only for those few candidates, a real 5-minute kline is fetched
     (`/api/v3/klines`) - kline volume is the true volume traded in that
     exact window, not a rolling estimate - to confirm the volume spike
     before alerting. This keeps API usage low (a handful of calls per
     cycle, not 480) while giving accurate volume data.

SETUP (local testing)
----------------------
1. pip install requests flask
2. Create a Telegram bot via @BotFather, get the bot token.
3. Get your chat_id: message your bot once, then visit
   https://api.telegram.org/bot<TOKEN>/getUpdates and read "chat":{"id":...}
4. Set the two environment variables below (or edit the constants directly):
     export TG_BOT_TOKEN="123456:ABC-your-token"
     export TG_CHAT_ID="123456789"
5. Run: python3 stakan_style_alerts.py
   This starts a small web server (health check for hosting platforms)
   plus the bot loop in a background thread.

Adjust the THRESHOLDS section to taste.
"""

import os
import time
import logging
import requests
from collections import defaultdict, deque

# ----------------------------- CONFIG ------------------------------------

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

CHECK_INTERVAL_SECONDS = 30               # poll every 30 seconds

# ----- FLIP / CONTINUATION SIGNAL -----
FLIP_LOOKBACK_MINUTES = 30                 # how far back to judge the prior trend
MIN_TREND_PCT_FOR_CANDIDATE = 1.0          # ignore noise: trend must move at least this % before checking volume
FLIP_VOLUME_MULTIPLIER = 2.0               # last 1h real volume vs the 1h just before it, for reversal alerts
CONTINUATION_VOLUME_MULTIPLIER = 3.0       # stricter volume bar for continuation alerts specifically

ALERT_COOLDOWN_SECONDS = 30 * 60           # don't re-alert same symbol+reason within this window
KLINE_INTERVAL = "5m"                      # candle size used for the real-volume check
KLINE_CANDLES_NEEDED = 24                  # 24 x 5m = 2h of candles (last 1h vs previous 1h)

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("stakan_alerts")

# ----------------------------- STATE --------------------------------------

# price history: symbol -> deque of (timestamp, price)
price_history = defaultdict(lambda: deque(maxlen=(FLIP_LOOKBACK_MINUTES * 60) // CHECK_INTERVAL_SECONDS + 2))

# last alert time per (symbol, reason) to avoid spam
last_alert_time = defaultdict(lambda: 0)


# ----------------------------- HELPERS -------------------------------------

def binance_get(url, params=None, timeout=15):
    """
    Wrapper around requests.get that respects Binance's rate-limit /
    IP-ban responses (429 = rate limited, 418 = IP auto-banned). Both
    include a Retry-After header telling us how long to wait. Raises the
    original HTTPError for anything else so callers can still handle it.
    """
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code in (418, 429):
        retry_after = int(resp.headers.get("Retry-After", 60))
        log.warning(
            "Binance returned %s (rate limited / IP ban). Waiting %ss before retrying.",
            resp.status_code, retry_after,
        )
        time.sleep(retry_after)
        resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def get_usdt_symbols():
    """Fetch all actively trading spot USDT pairs."""
    resp = binance_get(BINANCE_EXCHANGE_INFO_URL, timeout=15)
    data = resp.json()
    symbols = set()
    for s in data["symbols"]:
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s["isSpotTradingAllowed"]
        ):
            symbols.add(s["symbol"])
    return symbols


def fetch_all_tickers():
    """One call returns 24hr stats for every symbol (used for price only)."""
    resp = binance_get(BINANCE_TICKER_URL, timeout=20)
    return resp.json()


def fetch_real_volume_multiplier(symbol: str):
    """
    Real volume check using actual candle data (not a rolling-window
    estimate). Returns (multiplier, last_1h_volume, prev_1h_volume) or
    (None, None, None) if unavailable.
    """
    try:
        resp = binance_get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": KLINE_CANDLES_NEEDED},
            timeout=10,
        )
        candles = resp.json()
    except requests.RequestException as e:
        log.warning("Kline fetch failed for %s: %s", symbol, e)
        return None, None, None

    if len(candles) < KLINE_CANDLES_NEEDED:
        return None, None, None

    volumes = [float(c[5]) for c in candles]
    half = KLINE_CANDLES_NEEDED // 2
    prev_1h_volume = sum(volumes[:half])
    last_1h_volume = sum(volumes[half:])

    if prev_1h_volume <= 0:
        return None, last_1h_volume, prev_1h_volume

    return last_1h_volume / prev_1h_volume, last_1h_volume, prev_1h_volume


def send_telegram_alert(message: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram not configured, would have sent: %s", message)
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            log.error("Telegram send failed: %s", r.text)
    except requests.RequestException as e:
        log.error("Telegram send error: %s", e)


def can_alert(symbol: str, reason: str) -> bool:
    key = f"{symbol}:{reason}"
    now = time.time()
    if now - last_alert_time[key] >= ALERT_COOLDOWN_SECONDS:
        last_alert_time[key] = now
        return True
    return False


# ----------------------------- CORE LOGIC ----------------------------------

def process_cycle(symbols):
    now = time.time()
    tickers = fetch_all_tickers()
    alerts = []
    diagnostics = []  # (symbol, trend_pct, trend_sign, momentum_sign)

    candidates = []

    for t in tickers:
        symbol = t["symbol"]
        if symbol not in symbols:
            continue

        try:
            last_price = float(t["lastPrice"])
        except (KeyError, ValueError):
            continue
        if last_price <= 0:
            continue

        hist = price_history[symbol]
        hist.append((now, last_price))
        hist_list = list(hist)

        if len(hist_list) < 3:
            continue

        prev_price = hist_list[-2][1]
        window_start_price = hist_list[0][1]
        if window_start_price <= 0 or prev_price <= 0:
            continue

        momentum_change = last_price - prev_price
        trend_change = prev_price - window_start_price
        momentum_sign = 1 if momentum_change > 0 else (-1 if momentum_change < 0 else 0)
        trend_sign = 1 if trend_change > 0 else (-1 if trend_change < 0 else 0)
        trend_pct = trend_change / window_start_price * 100
        momentum_pct = momentum_change / prev_price * 100

        diagnostics.append((symbol, trend_pct, trend_sign, momentum_sign))

        if abs(trend_pct) < MIN_TREND_PCT_FOR_CANDIDATE:
            continue
        if trend_sign == 0 or momentum_sign == 0:
            continue

        window_prices = [px for _, px in hist_list]
        candidates.append({
            "symbol": symbol,
            "last_price": last_price,
            "trend_sign": trend_sign,
            "momentum_sign": momentum_sign,
            "trend_pct": trend_pct,
            "momentum_pct": momentum_pct,
            "window_high": max(window_prices),
            "window_low": min(window_prices),
        })

    for c in candidates:
        symbol = c["symbol"]
        multiplier, last_1h_vol, prev_1h_vol = fetch_real_volume_multiplier(symbol)
        if multiplier is None:
            continue

        is_reversal = c["trend_sign"] != c["momentum_sign"]
        is_continuation = c["trend_sign"] == c["momentum_sign"]

        range_str = (
            f"{FLIP_LOOKBACK_MINUTES}m range {c['window_low']:g}-{c['window_high']:g}, "
            f"trend {c['trend_pct']:+.2f}%, last tick {c['momentum_pct']:+.2f}%, "
            f"volume {multiplier:.1f}x vs prior hour"
        )

        if is_reversal and multiplier >= FLIP_VOLUME_MULTIPLIER:
            if c["trend_sign"] > 0 and can_alert(symbol, "flip_sell"):
                alerts.append(
                    f"🔴 SELL <b>{symbol}</b>: green→red flip after trending up over "
                    f"{FLIP_LOOKBACK_MINUTES}m (now {c['last_price']:g})\n{range_str}"
                )
            elif c["trend_sign"] < 0 and can_alert(symbol, "flip_buy"):
                alerts.append(
                    f"🟢 BUY <b>{symbol}</b>: red→green flip after trending down over "
                    f"{FLIP_LOOKBACK_MINUTES}m (now {c['last_price']:g})\n{range_str}"
                )
        elif is_continuation and multiplier >= CONTINUATION_VOLUME_MULTIPLIER:
            if c["trend_sign"] > 0 and can_alert(symbol, "continue_up"):
                alerts.append(
                    f"🟩 CONTINUATION UP <b>{symbol}</b>: still rising after "
                    f"{FLIP_LOOKBACK_MINUTES}m uptrend (now {c['last_price']:g})\n{range_str}"
                )
            elif c["trend_sign"] < 0 and can_alert(symbol, "continue_down"):
                alerts.append(
                    f"🟥 CONTINUATION DOWN <b>{symbol}</b>: still falling after "
                    f"{FLIP_LOOKBACK_MINUTES}m downtrend (now {c['last_price']:g})\n{range_str}"
                )

    return alerts, diagnostics


bot_status = {"state": "starting", "symbols_monitored": 0, "last_cycle": None, "last_error": None}


def run_bot_loop():
    symbols = None
    while symbols is None:
        try:
            log.info("Fetching tradable USDT pairs...")
            symbols = get_usdt_symbols()
            log.info("Monitoring %d USDT pairs", len(symbols))
            bot_status["symbols_monitored"] = len(symbols)
        except Exception as e:
            log.exception("Startup failed, retrying in 10s: %s", e)
            bot_status["last_error"] = str(e)
            time.sleep(10)

    bot_status["state"] = "running"

    if TG_BOT_TOKEN and TG_CHAT_ID:
        send_telegram_alert(f"✅ Alert bot started, monitoring {len(symbols)} USDT pairs.")
    else:
        log.warning(
            "TG_BOT_TOKEN / TG_CHAT_ID not set - alerts will only be logged, not sent."
        )

    while True:
        cycle_start = time.time()
        try:
            alerts, diagnostics = process_cycle(symbols)
            bot_status["last_cycle"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            for a in alerts:
                log.info("ALERT: %s", a)
                send_telegram_alert(a)
            if not alerts:
                if diagnostics:
                    top = sorted(diagnostics, key=lambda d: abs(d[1]), reverse=True)[:3]
                    top_str = ", ".join(
                        f"{sym} trend {pct:+.2f}%"
                        f"{' up' if trend > 0 else (' down' if trend < 0 else ' flat')}"
                        f"->{'up' if mom > 0 else ('down' if mom < 0 else 'flat')}"
                        for sym, pct, trend, mom in top
                    )
                    log.info("Cycle complete, no alerts. Closest: %s", top_str)
                else:
                    log.info("Cycle complete, no alerts (still building history).")
        except requests.RequestException as e:
            log.error("Network/API error this cycle: %s", e)
            bot_status["last_error"] = str(e)
        except Exception as e:
            log.exception("Unexpected error: %s", e)
            bot_status["last_error"] = str(e)

        elapsed = time.time() - cycle_start
        sleep_for = max(CHECK_INTERVAL_SECONDS - elapsed, 5)
        time.sleep(sleep_for)


# ----------------------------- WEB WRAPPER ---------------------------------
# Render's free tier only keeps "web services" alive (things that answer
# HTTP requests) - plain background scripts aren't supported on the free
# plan. This tiny Flask app exists purely so Render treats the bot as a web
# service; the actual bot logic runs in a background thread. Pair this with
# a free uptime pinger (e.g. UptimeRobot) hitting "/" every ~10 minutes so
# Render never puts it to sleep.

def create_app():
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route("/")
    def health():
        return jsonify(bot_status), 200

    return app


if __name__ == "__main__":
    import threading

    threading.Thread(target=run_bot_loop, daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app = create_app()
    app.run(host="0.0.0.0", port=port)
