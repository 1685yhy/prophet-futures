"""
历史记忆库构建器 — 从 akshare 拉取历史行情，逐日计算特征向量写入记忆库。
用法：
    from tools.history_builder import build_historical_memory
    count = build_historical_memory(['rb','lh','sc'], '20230101', '20251231')
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

from tools.memory_store import init_vector_db, compute_embedding, add_market_snapshot
from tools.indicators import calc_indicators

logger = logging.getLogger(__name__)

WINDOW = 60        # 计算指标所需的历史窗口
LABEL_WINDOW = 5   # 后续 N 日收益作为标签


def _fetch_kline(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从 akshare 拉取主力连续合约日线，返回标准列 DataFrame。"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(
            symbol=symbol.upper() + "0",
            start_date=start_date,
            end_date=end_date,
        )
        df.columns = ["date", "open", "high", "low", "close", "volume", "oi", "settle"]
        df["date"]  = df["date"].astype(str)
        df["close"] = df["close"].astype(float)
        df["open"]  = df["open"].astype(float)
        df["high"]  = df["high"].astype(float)
        df["low"]   = df["low"].astype(float)
        df["volume"]= df["volume"].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("Failed to fetch kline for %s: %s", symbol, e)
        return None


def _ma_label(ind: dict) -> str:
    if ind["ma5"] > ind["ma20"] > ind["ma60"]:
        return "多头排列"
    if ind["ma5"] < ind["ma20"] < ind["ma60"]:
        return "空头排列"
    return "均线纠缠"


def build_historical_memory(
    symbols: List[str],
    start_date: str = "20230101",
    end_date: Optional[str] = None,
    label_window: int = LABEL_WINDOW,
    db_path: str = "./vector_db",
) -> int:
    """
    拉取历史行情，每日构建特征向量和后续收益标签，写入记忆库。

    Args:
        symbols:      品种代码列表，如 ['rb', 'i', 'lh', 'jd', 'sc']
        start_date:   起始日期 'YYYYMMDD'
        end_date:     结束日期，默认今日
        label_window: 后续 N 日收益作为标签
        db_path:      向量库路径

    Returns:
        写入的总条数
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    init_vector_db(db_path)
    total_count = 0

    for symbol in symbols:
        logger.info("Building memory for %s (%s → %s)...", symbol, start_date, end_date)
        df = _fetch_kline(symbol, start_date, end_date)
        if df is None or len(df) < WINDOW + label_window + 1:
            logger.warning("Insufficient data for %s, skipping", symbol)
            continue

        symbol_count = 0
        for i in range(WINDOW, len(df) - label_window):
            window_df = df.iloc[i - WINDOW: i].copy()
            window_df = window_df.rename(columns={"oi": "open_interest"})

            try:
                ind = calc_indicators(window_df)
            except Exception as e:
                logger.debug("Indicator calc failed at index %d: %s", i, e)
                continue

            vec = compute_embedding(ind)

            # 后续 N 日收益（用 close 计算，百分比）
            entry_close  = float(df.iloc[i]["close"])
            exit_close   = float(df.iloc[i + label_window]["close"])
            future_ret   = (exit_close - entry_close) / (entry_close + 1e-8) * 100

            date_str = str(df.iloc[i]["date"])
            metadata = {
                "date":                  date_str,
                "symbol":                symbol,
                "description": (
                    f"{symbol} {date_str}: {_ma_label(ind)}, "
                    f"RSI={ind['rsi14']:.0f}, ADX={ind['adx14']:.0f}, "
                    f"ATR={ind['atr14']:.1f}, MACD_hist={ind['macd_hist']:.2f}"
                ),
                "subsequent_5d_return":  round(float(future_ret), 3),
                "ma_arrangement":        _ma_label(ind),
                "rsi":                   round(ind["rsi14"], 1),
                "adx":                   round(ind["adx14"], 1),
            }

            try:
                add_market_snapshot(vec, metadata)
                symbol_count += 1
            except Exception as e:
                logger.debug("Failed to add snapshot: %s", e)
                continue

        logger.info("  %s: %d records written", symbol, symbol_count)
        total_count += symbol_count

    logger.info("Memory build complete: %d total records", total_count)
    return total_count


def get_memory_stats(db_path: str = "./vector_db") -> dict:
    """返回记忆库统计信息。"""
    from pathlib import Path
    import json

    jsonl = Path(db_path) / "memories.jsonl"
    if not jsonl.exists():
        try:
            import chromadb
            client     = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection("market_memories")
            count      = collection.count()
            return {"backend": "chromadb", "total_records": count, "db_path": db_path}
        except Exception:
            return {"backend": "unknown", "total_records": 0}

    count = 0
    symbols: set = set()
    with open(jsonl) as f:
        for line in f:
            try:
                r = json.loads(line)
                count += 1
                symbols.add(r.get("meta", {}).get("symbol", "unknown"))
            except json.JSONDecodeError:
                continue
    return {
        "backend":       "jsonl",
        "total_records": count,
        "symbols":       sorted(symbols),
        "db_path":       db_path,
    }
