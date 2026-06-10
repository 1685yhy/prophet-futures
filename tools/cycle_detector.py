"""
大周期检测模块 — 判断品种所处的趋势周期。

核心逻辑（基于LH历史回测验证）：
- BULL：近120日上涨>5% + MA20>MA60 → 只允许做多
- BEAR：近120日下跌>5% + MA20<MA60 → 只允许做空
- NEUTRAL：震荡，不出信号

附加：换仓噪音检测（生猪合约换月期OI剧烈波动）
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Literal


CycleType = Literal["BULL", "BEAR", "NEUTRAL"]


def detect_cycle(df_window: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    """
    判断品种大周期。

    Args:
        df_window: 含 open/high/low/close/volume/oi 列的 DataFrame（至少 60 行）
        lookback:  回看天数，默认 120 日（约半年）

    Returns:
        {
            "cycle":        "BULL" | "BEAR" | "NEUTRAL",
            "ret_6m":       float,   # 近 lookback 日收益率
            "ma20_vs_ma60": float,   # MA20/MA60 - 1
            "strength":     float,   # 0-1，趋势强度
            "reasoning":    str,
        }
    """
    closes = df_window["close"].values.astype(float)
    if len(closes) < 60:
        return {"cycle": "NEUTRAL", "ret_6m": 0.0, "ma20_vs_ma60": 0.0,
                "strength": 0.0, "reasoning": "数据不足"}

    lb      = min(lookback, len(closes) - 1)
    ret_6m  = (closes[-1] - closes[-lb]) / (closes[-lb] + 1e-8)
    ma20    = float(np.mean(closes[-20:]))
    ma60    = float(np.mean(closes[-min(60, len(closes)):]))
    ma_ratio= ma20 / (ma60 + 1e-8) - 1.0

    # 趋势强度：收益率和均线偏离的几何均值
    strength = min(1.0, (abs(ret_6m) * 10 + abs(ma_ratio) * 20) / 2)

    if ma20 > ma60 and ret_6m > 0.05:
        cycle     = "BULL"
        reasoning = f"MA20({ma20:.0f})>MA60({ma60:.0f})，{lb}日涨幅{ret_6m:.1%}"
    elif ma20 < ma60 and ret_6m < -0.05:
        cycle     = "BEAR"
        reasoning = f"MA20({ma20:.0f})<MA60({ma60:.0f})，{lb}日跌幅{ret_6m:.1%}"
    else:
        cycle     = "NEUTRAL"
        strength  = 0.0
        reasoning = f"震荡：MA偏离{ma_ratio:.1%}，{lb}日变化{ret_6m:.1%}"

    return {
        "cycle":        cycle,
        "ret_6m":       round(ret_6m, 4),
        "ma20_vs_ma60": round(ma_ratio, 4),
        "strength":     round(strength, 3),
        "reasoning":    reasoning,
    }


def detect_rollover_noise(df_window: pd.DataFrame) -> Dict[str, Any]:
    """
    检测换仓期OI噪音（生猪、鸡蛋等农产品合约换月时OI剧烈变化）。

    规则：
    - 计算近10日OI日变化的标准差
    - 若今日OI变化 > 3×标准差，视为换仓噪音
    - 换仓噪音期间不应用单日OI方向判断

    Returns:
        {
            "is_noise":      bool,
            "oi_today_chg":  float,
            "oi_std_10d":    float,
            "noise_ratio":   float,   # 今日变化/标准差
        }
    """
    if "oi" not in df_window.columns:
        return {"is_noise": False, "oi_today_chg": 0.0, "oi_std_10d": 0.0, "noise_ratio": 0.0}

    oi = df_window["oi"].values.astype(float)
    if len(oi) < 11:
        return {"is_noise": False, "oi_today_chg": 0.0, "oi_std_10d": 0.0, "noise_ratio": 0.0}

    oi_diffs    = np.diff(oi[-11:])   # 近10日变化量
    oi_std      = float(np.std(oi_diffs[:-1]))  # 前9日的标准差
    oi_today    = float(oi_diffs[-1])            # 今日变化
    noise_ratio = abs(oi_today) / (oi_std + 1e-8)
    is_noise    = noise_ratio > 3.0

    return {
        "is_noise":     is_noise,
        "oi_today_chg": round(oi_today, 0),
        "oi_std_10d":   round(oi_std, 0),
        "noise_ratio":  round(noise_ratio, 2),
    }


def get_lh_signal_conditions(
    df_window: pd.DataFrame,
    ind: dict,
    fundamentals: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    生猪(LH)专项高胜率信号判断（v2，含基本面）。
    基于4年历史回测验证的条件组合：

    做空（80% 5日准确率）：
        大周期BEAR + MA空头 + MACD负不收窄 + OI减/FLAT + ADX>20 + 非换仓噪音
        基本面增强：基差升水高 → 置信度+4-8%
        基本面否决：基本面综合偏多(+2)时不做空

    做多（61% 5日准确率，样本少）：
        大周期BULL + MA多头 + MACD正 + OI积累 + ADX>22 + RSI30-65 + 非换仓噪音
        基本面否决：基差升水>20%时禁止做多（期货透支现货）

    Args:
        df_window:    K线 DataFrame
        ind:          技术指标字典
        fundamentals: get_hog_fundamentals() 的返回值（可选，有则使用基本面加成）

    Returns:
        {
            "signal":          "SHORT" | "LONG" | "WAIT",
            "confidence":      float,
            "conditions":      dict,
            "stop_atr_mult":   float,
            "target_atr_mult": float,
            "hold_days":       int,
            "reasoning":       str,
        }
    """
    from tools.indicators import _calc_macd

    closes = df_window["close"].values.astype(float)
    cycle_info   = detect_cycle(df_window)
    noise_info   = detect_rollover_noise(df_window)
    cycle        = cycle_info["cycle"]

    # MACD 是否在负值区收窄（动能好转，不适合做空）
    _, _, h0 = _calc_macd(closes)
    _, _, h1 = _calc_macd(closes[:-1]) if len(closes) > 1 else (0, 0, 0)
    _, _, h2 = _calc_macd(closes[:-2]) if len(closes) > 2 else (0, 0, 0)
    macd_improving = bool(h0 < 0 and h1 < 0 and abs(h0) < abs(h1) < abs(h2))

    adx  = ind.get("adx14", 0)
    rsi  = ind.get("rsi14", 50)
    ma5  = ind.get("ma5",  0)
    ma20 = ind.get("ma20", 0)
    ma60 = ind.get("ma60", 0)
    macd_h = ind.get("macd_hist", 0)

    oi = df_window["oi"].values.astype(float) if "oi" in df_window.columns else np.zeros(10)
    oi_3d = float(oi[-1] - oi[-4]) if len(oi) >= 4 else 0
    oi_5d = float(oi[-1] - oi[-6]) if len(oi) >= 6 else oi_3d
    if   oi_3d > 0 and oi_5d > 0: oi_trend = "ACCUMULATING"
    elif oi_3d < 0 and oi_5d < 0: oi_trend = "REDUCING"
    else:                           oi_trend = "FLAT"

    ma_bull = ma5 > ma20 > ma60
    ma_bear = ma5 < ma20 < ma60

    conditions = {
        "cycle":           cycle,
        "ma_bull":         ma_bull,
        "ma_bear":         ma_bear,
        "macd_pos":        macd_h > 0,
        "macd_neg":        macd_h < 0 and not macd_improving,
        "macd_improving":  macd_improving,
        "oi_trend":        oi_trend,
        "adx_ok_long":     adx > 22,
        "adx_ok_short":    adx > 20,
        "rsi_ok_long":     30 < rsi < 65,
        "rsi_ok_short":    32 < rsi < 72,
        "no_noise":        not noise_info["is_noise"],
    }

    # ── 基本面数据（选填，有则使用）────────────────────────────────────────
    # 通过 fundamentals 参数传入（由 get_hog_fundamentals() 产生）
    # 无则用技术面单独判断
    basis_pct      = fundamentals.get("basis_pct", 0) if fundamentals else 0
    basis_signal   = fundamentals.get("basis_signal", "NORMAL") if fundamentals else "NORMAL"
    spot_trend     = fundamentals.get("spot_trend", "FLAT") if fundamentals else "FLAT"
    fund_score     = fundamentals.get("fundamental_score", 0) if fundamentals else 0
    profit_signal  = fundamentals.get("profit_signal", "BREAKEVEN") if fundamentals else "BREAKEVEN"
    supply_signal  = fundamentals.get("supply_signal", "STABLE") if fundamentals else "STABLE"

    conditions["basis_signal"]  = basis_signal
    conditions["spot_trend"]    = spot_trend
    conditions["fund_score"]    = fund_score

    # ── 基本面否决规则（动态阈值版）───────────────────────────────────────
    # 规则1：基差超过该合约历史90%分位（EXTREME_PREMIUM）才否决做多
    # 注：2609合约历史均值22.5%，90%分位32.3%——25%是正常不否决
    extreme_premium_veto_long = basis_signal == "EXTREME_PREMIUM"

    # 规则2：现货持续下跌 + 基差比历史均值高很多（>15%偏差），做多降权
    basis_vs_mean = fundamentals.get("basis_vs_mean", 0) if fundamentals else 0
    spot_falling_veto_long = spot_trend == "FALLING" and basis_vs_mean > 15

    # 规则3：基本面综合偏多（+2以上）时，做空信号降权
    fundamental_bullish_veto_short = fund_score >= 2

    # ── 做空信号（80%历史准确率）──
    short_ok = (
        cycle == "BEAR"
        and ma_bear
        and conditions["macd_neg"]
        and oi_trend in ("REDUCING", "FLAT")
        and conditions["adx_ok_short"]
        and conditions["rsi_ok_short"]
        and conditions["no_noise"]
        and not fundamental_bullish_veto_short   # 基本面偏多时不做空
    )

    # ── 做多信号（61%历史准确率，样本少）──
    long_ok = (
        cycle == "BULL"
        and ma_bull
        and conditions["macd_pos"]
        and oi_trend == "ACCUMULATING"
        and conditions["adx_ok_long"]
        and conditions["rsi_ok_long"]
        and conditions["no_noise"]
        and not extreme_premium_veto_long    # 基差极度升水时禁止做多
        and not spot_falling_veto_long       # 现货跌+高升水时禁止做多
    )

    # ── 基本面加成置信度 ────────────────────────────────────────────────
    basis_boost = 0.0
    if short_ok:
        # 基差极度升水 → 做空置信度提升
        if basis_signal == "EXTREME_PREMIUM":
            basis_boost = 0.08
        elif basis_signal == "HIGH_PREMIUM":
            basis_boost = 0.04
        # 现货下跌 → 做空置信度提升
        if spot_trend == "FALLING":
            basis_boost += 0.04
        # 供给扩张 → 做空置信度提升
        if supply_signal == "EXPANDING":
            basis_boost += 0.03

    fund_reasoning = ""
    if fundamentals:
        fund_reasoning = (f" | 基差{basis_pct:.1f}%({basis_signal})"
                          f" 现货{spot_trend} 基本面评分{fund_score:+d}")

    if short_ok:
        return {
            "signal":          "SHORT",
            "confidence":      min(0.92, 0.70 + cycle_info["strength"] * 0.15 + basis_boost),
            "conditions":      conditions,
            "stop_atr_mult":   1.5,
            "target_atr_mult": 2.5,
            "hold_days":       5,
            "reasoning":       f"LH做空80%条件满足: {cycle_info['reasoning']}{fund_reasoning}",
        }
    elif long_ok:
        return {
            "signal":          "LONG",
            "confidence":      min(0.70, 0.55 + cycle_info["strength"] * 0.15),
            "conditions":      conditions,
            "stop_atr_mult":   1.5,
            "target_atr_mult": 2.5,
            "hold_days":       5,
            "reasoning":       f"LH做多61%条件满足: {cycle_info['reasoning']}{fund_reasoning}",
        }
    else:
        veto_reasons = []
        if extreme_premium_veto_long: veto_reasons.append(f"基差升水{basis_pct:.1f}%否决做多")
        if spot_falling_veto_long:    veto_reasons.append(f"现货下跌+高升水否决做多")
        if fundamental_bullish_veto_short: veto_reasons.append("基本面偏多否决做空")
        veto_str = " | ".join(veto_reasons) if veto_reasons else ""

        return {
            "signal":          "WAIT",
            "confidence":      0.0,
            "conditions":      conditions,
            "stop_atr_mult":   0.0,
            "target_atr_mult": 0.0,
            "hold_days":       0,
            "reasoning":       (f"条件不满足: cycle={cycle}, ma_bear={ma_bear}, "
                                f"macd_neg={conditions['macd_neg']}"
                                + (f" | 否决: {veto_str}" if veto_str else "")),
        }


def get_generic_signal_conditions(
    symbol: str, df_window: pd.DataFrame, ind: dict
) -> Dict[str, Any]:
    """
    通用品种信号判断（JD、FU/BU、MA等）。
    使用与LH相同的框架但置信度较低。
    """
    from tools.indicators import _calc_macd

    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)
    cycle      = cycle_info["cycle"]

    _, _, h0 = _calc_macd(closes)
    _, _, h1 = _calc_macd(closes[:-1]) if len(closes) > 1 else (0, 0, 0)
    _, _, h2 = _calc_macd(closes[:-2]) if len(closes) > 2 else (0, 0, 0)
    macd_improving = bool(h0 < 0 and h1 < 0 and abs(h0) < abs(h1) < abs(h2))

    adx = ind.get("adx14", 0); rsi = ind.get("rsi14", 50)
    ma5 = ind.get("ma5", 0); ma20 = ind.get("ma20", 0); ma60 = ind.get("ma60", 0)
    macd_h = ind.get("macd_hist", 0)
    ma_bull = ma5 > ma20 > ma60; ma_bear = ma5 < ma20 < ma60

    oi = df_window["oi"].values.astype(float) if "oi" in df_window.columns else np.zeros(10)
    oi_3d = float(oi[-1] - oi[-4]) if len(oi) >= 4 else 0
    oi_5d = float(oi[-1] - oi[-6]) if len(oi) >= 6 else oi_3d
    if   oi_3d > 0 and oi_5d > 0: oi_trend = "ACCUMULATING"
    elif oi_3d < 0 and oi_5d < 0: oi_trend = "REDUCING"
    else:                           oi_trend = "FLAT"

    short_ok = (cycle == "BEAR" and ma_bear and macd_h < 0 and not macd_improving
                and oi_trend in ("REDUCING", "FLAT") and adx > 22 and not noise_info["is_noise"])
    long_ok  = (cycle == "BULL" and ma_bull and macd_h > 0
                and oi_trend == "ACCUMULATING" and adx > 22 and 30 < rsi < 65
                and not noise_info["is_noise"])

    if short_ok:
        return {"signal": "SHORT", "confidence": 0.62, "stop_atr_mult": 1.5,
                "target_atr_mult": 2.5, "hold_days": 5,
                "reasoning": f"{symbol}做空: {cycle_info['reasoning']}"}
    elif long_ok:
        return {"signal": "LONG", "confidence": 0.58, "stop_atr_mult": 1.5,
                "target_atr_mult": 2.5, "hold_days": 5,
                "reasoning": f"{symbol}做多: {cycle_info['reasoning']}"}
    return {"signal": "WAIT", "confidence": 0.0, "stop_atr_mult": 0.0,
            "target_atr_mult": 0.0, "hold_days": 0,
            "reasoning": f"{symbol} 条件不足: cycle={cycle}"}
