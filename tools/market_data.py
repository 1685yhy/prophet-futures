"""Market data retrieval — wraps akshare with fallback to synthetic data for testing."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from models.schemas import KlineData, MarketSnapshot

logger = logging.getLogger(__name__)

FUTURES_SYMBOLS = ["rb", "i", "j", "jm", "hc", "sc", "au", "ag", "cu", "al",
                   "zn", "ni", "bu", "ru", "pg", "eg", "sp", "ss", "lh", "jd"]


def get_contracts(active_only: bool = True) -> List[str]:
    try:
        import akshare as ak
        df = ak.futures_display_main_sina()
        return list(df["symbol"].values[:20]) if not df.empty else FUTURES_SYMBOLS[:10]
    except Exception as e:
        logger.warning("akshare unavailable, using default symbols: %s", e)
        return FUTURES_SYMBOLS[:10]


def get_kline(symbol: str, interval: str = "daily", periods: int = 120) -> KlineData:
    try:
        import akshare as ak
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=periods * 2)).strftime("%Y%m%d")
        df    = ak.futures_main_sina(symbol=symbol.upper() + "0", start_date=start, end_date=end)
        df    = df.tail(periods).reset_index(drop=True)
        col_map = {"日期":"date","开盘价":"open","最高价":"high",
                   "最低价":"low","收盘价":"close","成交量":"volume",
                   "持仓量":"oi","动态结算价":"settle"}
        df = df.rename(columns=col_map)
        kline = KlineData(
            symbol=symbol, interval=interval,
            timestamps=df["date"].astype(str).tolist(),
            opens=df["open"].tolist(), highs=df["high"].tolist(),
            lows=df["low"].tolist(),   closes=df["close"].tolist(),
            volumes=df["volume"].tolist(),
            open_interests=df["oi"].tolist() if "oi" in df.columns else None,
        )
        # 结算价序列供信号计算使用（行业标准：用结算价而非收盘价）
        if "settle" in df.columns:
            kline._settles = [float(v) for v in df["settle"].tolist()]
        return kline
    except Exception as e:
        logger.warning("API error for %s, generating synthetic data: %s", symbol, e)
        return _synthetic_kline(symbol, interval, periods)


def get_realtime_quote(symbol: str) -> MarketSnapshot:
    try:
        import akshare as ak
        df    = ak.futures_zh_spot(symbol=symbol.upper() + "0", market="CF", adjust="0")
        row   = df.iloc[0]
        price = float(row.get("current_price", 0) or 5000.0)
        last_settle = float(row.get("last_settle_price", 0) or price)
        change_pct  = round((price - last_settle) / (last_settle + 1e-8) * 100, 2) if last_settle else 0.0
        return MarketSnapshot(
            symbol=symbol, last_price=price,
            bid=float(row.get("bid_price", price - 1)),
            ask=float(row.get("ask_price", price + 1)),
            volume=float(row.get("volume", 0)),
            open_interest=float(row.get("hold", 0)),
            change_pct=change_pct,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.warning("Realtime quote unavailable for %s: %s", symbol, e)
        return _synthetic_quote(symbol)


def get_tick_data(symbol: str, lookback_seconds: int = 60) -> List[dict]:
    try:
        import akshare as ak
        df = ak.futures_zh_minute_sina(symbol=symbol + "0", period="1")
        return df.tail(lookback_seconds // 60 + 5).to_dict("records")
    except Exception as e:
        logger.warning("Tick data unavailable: %s", e)
        base = 5000.0
        return [
            {"time": (datetime.now() - timedelta(seconds=i)).isoformat(),
             "price": base + np.random.normal(0, 5),
             "volume": int(np.random.exponential(100))}
            for i in range(min(lookback_seconds, 60), 0, -5)
        ]


def plot_kline_chart(symbol: str, interval: str = "daily", periods: int = 60) -> Optional[bytes]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from tools.indicators import calc_indicators

        kline = get_kline(symbol, interval, periods)
        df    = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low": kline.lows,   "close": kline.closes, "volume": kline.volumes,
        })
        ind = calc_indicators(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"{symbol} — {interval}", fontsize=14)
        for i, (o, h, l, c) in enumerate(zip(df["open"], df["high"], df["low"], df["close"])):
            color = "red" if c >= o else "green"
            ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax1.bar(i, abs(c - o), bottom=min(o, c), color=color, width=0.6, alpha=0.8)
        ax1.axhline(ind["ma20"],     color="blue",  linewidth=1,   label="MA20")
        ax1.axhline(ind["bb_upper"], color="gray",  linewidth=0.8, linestyle="--")
        ax1.axhline(ind["bb_lower"], color="gray",  linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Price"); ax1.legend(fontsize=8)
        ax2.bar(range(len(kline.volumes)), kline.volumes, color="steelblue", alpha=0.7)
        ax2.set_ylabel("Volume")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        return buf.getvalue()
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)
        return None


def _synthetic_kline(symbol: str, interval: str, periods: int) -> KlineData:
    np.random.seed(hash(symbol) % 2**32)
    base   = 5000.0
    closes = [base]
    for _ in range(periods - 1):
        closes.append(max(100.0, closes[-1] * (1 + np.random.normal(0, 0.015))))
    opens, highs, lows = [], [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.005)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.005)))
        opens.append(round(o, 2)); highs.append(round(h, 2)); lows.append(round(l, 2))
    start      = datetime.now() - timedelta(days=periods)
    timestamps = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(periods)]
    volumes    = [float(int(np.random.exponential(10000))) for _ in range(periods)]
    return KlineData(
        symbol=symbol, interval=interval, timestamps=timestamps,
        opens=opens, highs=highs, lows=lows,
        closes=[round(c, 2) for c in closes], volumes=volumes,
    )


def _synthetic_quote(symbol: str) -> MarketSnapshot:
    np.random.seed(hash(symbol) % 2**32)
    price = round(5000 + np.random.normal(0, 500), 2)
    return MarketSnapshot(
        symbol=symbol, last_price=price,
        bid=price - 1, ask=price + 1,
        volume=float(int(np.random.exponential(10000))),
        open_interest=float(int(np.random.exponential(50000))),
        change_pct=round(np.random.normal(0, 1.5), 2),
        timestamp=datetime.now().isoformat(),
    )
