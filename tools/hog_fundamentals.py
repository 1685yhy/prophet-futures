"""
生猪产业链基本面数据模块

数据来源（akshare）：
  spot_hog_lean_price_soozhu   — 瘦肉型猪（外三元参考）现货价格，元/公斤
  spot_hog_three_way_soozhu    — 外三元价格，元/公斤（近期）
  spot_hog_crossbred_soozhu    — 二元猪价格，元/公斤
  spot_hog_soozhu              — 全国各省今日现货价
  spot_hog_year_trend_soozhu   — 全年现货价趋势
  index_hog_spot_price         — 猪价指数（日频）
  futures_hog_core             — 自繁自养完全成本
  futures_hog_cost             — 仔猪外购成本
  futures_hog_supply           — 能繁母猪存栏（月频）
  futures_spot_price           — 期现基差（日频，含LH）

输出：结构化的基本面画像字典，供其他模块直接调用
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_hog_fundamentals(date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    获取生猪完整基本面数据。

    Returns:
        {
            # 现货价格
            "spot_price":          float,   # 瘦肉型现货，元/公斤
            "spot_7d_change":      float,   # 7日涨跌幅（%）
            "spot_30d_change":     float,   # 30日涨跌幅（%）
            "spot_trend":          str,     # "RISING"|"FALLING"|"FLAT"
            "national_avg":        float,   # 全国均价，元/公斤
            "three_way_price":     float,   # 外三元，元/公斤

            # 期现基差
            "futures_price":       float,   # 期货主力结算价，元/吨
            "basis_pct":           float,   # 升贴水率（%）
            "basis_signal":        str,     # "EXTREME_PREMIUM"|"HIGH_PREMIUM"|"NORMAL"|"DISCOUNT"

            # 养殖成本与盈利
            "cost_per_kg":         float,   # 自繁自养完全成本，元/公斤
            "profit_margin":       float,   # 现货-成本（元/公斤）
            "profit_signal":       str,     # "PROFIT"|"BREAKEVEN"|"LOSS"

            # 供给（能繁母猪）
            "sow_inventory":       float,   # 百万头
            "sow_3m_change":       float,   # 近3月变化（百万头）
            "supply_signal":       str,     # "TIGHTENING"|"EXPANDING"|"STABLE"

            # 价格指数
            "price_index":         float,
            "index_vs_ma4":        float,   # 指数/4月均线（>1偏强，<1偏弱）
            "index_7d_change":     float,

            # 综合基本面评分（-3 到 +3）
            # +: 利多期货  -: 利空期货
            "fundamental_score":   int,
            "fundamental_signals": list,    # 触发的信号描述
        }
    """
    result = _empty_fundamentals()
    signals = []
    score = 0

    # ── 现货价格 ──────────────────────────────────────────────────────────
    try:
        import akshare as ak
        lean_df = ak.spot_hog_lean_price_soozhu()
        spot_now = float(lean_df.iloc[-1]["价格"])
        result["spot_price"] = spot_now

        if len(lean_df) >= 8:
            spot_7d = float(lean_df.iloc[-8]["价格"])
            result["spot_7d_change"] = round((spot_now - spot_7d) / spot_7d * 100, 2)

        # 全年走势
        trend_df = ak.spot_hog_year_trend_soozhu()
        if len(trend_df) >= 31:
            spot_30d = float(trend_df.iloc[-31]["价格"])
            result["spot_30d_change"] = round((spot_now - spot_30d) / spot_30d * 100, 2)

        chg7d = result["spot_7d_change"]
        if chg7d > 1.5:
            result["spot_trend"] = "RISING"
            score += 1
            signals.append(f"现货7日上涨{chg7d:.1f}% → 利多期货(+1)")
        elif chg7d < -1.5:
            result["spot_trend"] = "FALLING"
            score -= 1
            signals.append(f"现货7日下跌{chg7d:.1f}% → 利空期货(-1)")
        else:
            result["spot_trend"] = "FLAT"
            signals.append(f"现货7日平稳{chg7d:.1f}%")
    except Exception as e:
        logger.warning("现货价格获取失败: %s", e)

    # ── 外三元价格 ────────────────────────────────────────────────────────
    try:
        import akshare as ak
        three_df = ak.spot_hog_three_way_soozhu()
        result["three_way_price"] = float(three_df.iloc[-1]["价格"])
    except Exception as e:
        logger.warning("外三元价格获取失败: %s", e)

    # ── 全国均价 ──────────────────────────────────────────────────────────
    try:
        import akshare as ak
        prov_df = ak.spot_hog_soozhu()
        result["national_avg"] = round(float(prov_df["价格"].mean()), 2)
    except Exception as e:
        logger.warning("全国均价获取失败: %s", e)

    # ── 期现基差 ──────────────────────────────────────────────────────────
    try:
        import akshare as ak
        d = date_str or datetime.now().strftime("%Y%m%d")
        # 尝试今日，若失败退一天
        for days_back in [0, 1, 2, 3]:
            try:
                check_d = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                spot_df = ak.futures_spot_price(date=check_d)
                lh_row  = spot_df[spot_df["symbol"].str.upper() == "LH"]
                if not lh_row.empty:
                    row = lh_row.iloc[0]
                    result["futures_price"] = float(row.get("dominant_contract_price", 0))
                    basis_pct = float(row.get("dom_basis_rate", 0)) * 100
                    result["basis_pct"] = round(basis_pct, 2)
                    if result["spot_price"] == 0 and row.get("spot_price"):
                        result["spot_price"] = float(row["spot_price"]) / 1000  # 元/吨→元/公斤

                    # ── 按合约月份动态判断基差是否异常 ──────────────────
                    # 历史统计（各月份合约基差均值和90%分位）：
                    #   01月: 均-1.2%  90%=+4.0%   03月: 均-3.8%  90%=+1.7%
                    #   05月: 均+2.3%  90%=+7.8%   07月: 均+11.9% 90%=+19.8%
                    #   09月: 均+22.5% 90%=+32.3%  11月: 数据不足
                    # 结论：09月合约天然升水高，25%对2609是正常，不是"极端"
                    # 阈值 = 历史90%分位（超过才是真正异常）
                    CONTRACT_BASIS_90PCT = {
                        1: 4.0, 3: 1.7, 5: 7.8, 7: 19.8, 9: 32.3, 11: 20.0
                    }
                    CONTRACT_BASIS_MEAN = {
                        1: -1.2, 3: -3.8, 5: 2.3, 7: 11.9, 9: 22.5, 11: 8.0
                    }
                    from datetime import datetime as _dt
                    dom_contract = str(row.get("dominant_contract", "")) or ""
                    # 从主力合约代码提取月份（如 LH2609 → 9）
                    contract_month = None
                    for part in [dom_contract, str(row.get("near_contract",""))]:
                        if len(part) >= 6 and part[:2].upper() == "LH":
                            try:
                                contract_month = int(part[4:6])
                                break
                            except: pass
                    if contract_month is None:
                        contract_month = _dt.now().month + 3  # 估算
                        contract_month = ((contract_month - 1) // 2) * 2 + 1  # 奇数月
                        contract_month = min(contract_month, 11)

                    basis_mean   = CONTRACT_BASIS_MEAN.get(contract_month, 8.0)
                    basis_90pct  = CONTRACT_BASIS_90PCT.get(contract_month, 20.0)
                    basis_vs_mean= basis_pct - basis_mean  # 相对于历史均值的偏差

                    result["contract_month"]     = contract_month
                    result["basis_mean_hist"]    = basis_mean
                    result["basis_vs_mean"]      = round(basis_vs_mean, 1)

                    if basis_pct > basis_90pct:
                        # 真正超过历史90%分位，才算极端
                        result["basis_signal"] = "EXTREME_PREMIUM"
                        score -= 2
                        signals.append(
                            f"基差{basis_pct:.1f}%超过{contract_month}月合约历史90%分位({basis_90pct:.0f}%)，真正偏高(-2)")
                    elif basis_vs_mean > 8:
                        # 比历史均值高8%以上（中度偏高）
                        result["basis_signal"] = "HIGH_PREMIUM"
                        score -= 1
                        signals.append(
                            f"基差{basis_pct:.1f}%比{contract_month}月合约历史均值({basis_mean:.0f}%)高{basis_vs_mean:.0f}%，偏高(-1)")
                    elif basis_pct < basis_mean - 8:
                        # 比历史均值低8%以上（偏低，可能是利多）
                        result["basis_signal"] = "DISCOUNT"
                        score += 1
                        signals.append(
                            f"基差{basis_pct:.1f}%低于{contract_month}月合约历史均值({basis_mean:.0f}%)，偏低有修复空间(+1)")
                    else:
                        result["basis_signal"] = "NORMAL"
                        signals.append(
                            f"基差{basis_pct:.1f}%（{contract_month}月合约历史均值{basis_mean:.0f}%），正常范围")
                    break
            except Exception:
                continue
    except Exception as e:
        logger.warning("期现基差获取失败: %s", e)

    # ── 养殖成本与盈利 ────────────────────────────────────────────────────
    try:
        import akshare as ak
        core_df = ak.futures_hog_core()
        cost_per_kg = float(core_df.iloc[-1]["value"])
        result["cost_per_kg"] = cost_per_kg

        spot_kg = result["spot_price"]
        if spot_kg > 0:
            margin = round(spot_kg - cost_per_kg, 2)
            result["profit_margin"] = margin
            if margin > 1.0:
                result["profit_signal"] = "PROFIT"
                score += 1
                signals.append(f"养殖盈利{margin:.2f}元/公斤，扩栏意愿强 → 供给压力上升(-0，中性)")
            elif margin > -0.5:
                result["profit_signal"] = "BREAKEVEN"
                signals.append(f"养殖接近盈亏平衡({margin:.2f}元/公斤)")
            else:
                result["profit_signal"] = "LOSS"
                score += 1
                signals.append(f"养殖亏损{margin:.2f}元/公斤，出栏积极性下降 → 供给收缩利多(+1)")
    except Exception as e:
        logger.warning("养殖成本获取失败: %s", e)

    # ── 能繁母猪（供给端）────────────────────────────────────────────────
    try:
        import akshare as ak
        supply_df = ak.futures_hog_supply()
        result["sow_inventory"] = float(supply_df.iloc[-1]["value"])

        if len(supply_df) >= 4:
            chg_3m = float(supply_df.iloc[-1]["value"]) - float(supply_df.iloc[-4]["value"])
            result["sow_3m_change"] = round(chg_3m, 2)
            if chg_3m < -0.3:
                result["supply_signal"] = "TIGHTENING"
                score += 1
                signals.append(f"能繁母猪3月减少{chg_3m:.2f}百万头，6-9月供给收缩利多(+1)")
            elif chg_3m > 0.3:
                result["supply_signal"] = "EXPANDING"
                score -= 1
                signals.append(f"能繁母猪3月增加{chg_3m:.2f}百万头，6-9月供给扩张利空(-1)")
            else:
                result["supply_signal"] = "STABLE"
    except Exception as e:
        logger.warning("能繁母猪数据获取失败: %s", e)

    # ── 价格指数 ──────────────────────────────────────────────────────────
    try:
        import akshare as ak
        idx_df = ak.index_hog_spot_price()
        latest = idx_df.iloc[-1]
        result["price_index"]    = float(latest["指数"])
        result["index_vs_ma4"]   = round(float(latest["指数"]) / (float(latest["4个月均线"]) + 1e-8), 3)
        result["index_7d_change"]= round(float(latest["指数"]) - float(idx_df.iloc[-8]["指数"]), 2) if len(idx_df) >= 8 else 0

        if result["index_vs_ma4"] < 0.90:
            score -= 1
            signals.append(f"价格指数{result['price_index']:.1f}低于4月均线10%以上，市场偏弱(-1)")
        elif result["index_vs_ma4"] > 1.05:
            score += 1
            signals.append(f"价格指数{result['price_index']:.1f}高于4月均线5%以上，市场偏强(+1)")
    except Exception as e:
        logger.warning("价格指数获取失败: %s", e)

    result["fundamental_score"]   = max(-3, min(3, score))
    result["fundamental_signals"] = signals
    return result


def get_fundamental_verdict(fundamentals: Dict[str, Any]) -> Dict[str, str]:
    """
    将基本面评分转化为做多/做空的直接判断。

    Returns:
        {
            "direction":  "BULLISH"|"BEARISH"|"NEUTRAL",
            "strength":   "STRONG"|"MODERATE"|"WEAK",
            "summary":    str,  # 一句话总结
        }
    """
    score       = fundamentals.get("fundamental_score", 0)
    basis       = fundamentals.get("basis_signal", "NORMAL")
    basis_pct   = fundamentals.get("basis_pct", 0)
    basis_mean  = fundamentals.get("basis_mean_hist", 8.0)
    contract_m  = fundamentals.get("contract_month", 9)
    trend       = fundamentals.get("spot_trend", "FLAT")

    # EXTREME_PREMIUM = 真正超过该合约历史90%分位，才是强烈空头信号
    # 对于2609（历史均值22.5%，90%分位32.3%），25%是正常，不触发此逻辑
    if basis == "EXTREME_PREMIUM":
        return {
            "direction": "BEARISH",
            "strength":  "STRONG",
            "summary":   (f"基差{basis_pct:.1f}%超过{contract_m}月合约历史90%分位，"
                          f"相对历史均值({basis_mean:.0f}%)真正偏高，有回归压力"),
        }

    if score >= 2:
        return {"direction": "BULLISH", "strength": "STRONG" if score == 3 else "MODERATE",
                "summary": f"基本面偏多(得分+{score})：{fundamentals['fundamental_signals'][0] if fundamentals['fundamental_signals'] else ''}"}
    elif score <= -2:
        return {"direction": "BEARISH", "strength": "STRONG" if score == -3 else "MODERATE",
                "summary": f"基本面偏空(得分{score})：{fundamentals['fundamental_signals'][0] if fundamentals['fundamental_signals'] else ''}"}
    else:
        return {"direction": "NEUTRAL", "strength": "WEAK",
                "summary": f"基本面中性(得分{score})"}


def format_fundamentals_report(f: Dict[str, Any]) -> str:
    """生成基本面报告字符串，用于 daily_update 输出。"""
    v = get_fundamental_verdict(f)
    dir_cn = {"BULLISH":"偏多↑","BEARISH":"偏空↓","NEUTRAL":"中性→"}.get(v["direction"],"?")
    str_cn = {"STRONG":"强","MODERATE":"中","WEAK":"弱"}.get(v["strength"],"?")

    lines = ["【产业基本面】"]

    # 现货
    spot = f.get("spot_price", 0)
    fut  = f.get("futures_price", 0)
    basis= f.get("basis_pct", 0)
    if spot > 0:
        lines.append(f"  现货价: {spot:.2f}元/公斤  期货主力: {fut:.0f}元/吨  基差: {basis:+.1f}%")
        lines.append(f"  7日现货变化: {f.get('spot_7d_change',0):+.1f}%  趋势: {f.get('spot_trend','N/A')}")

    # 成本盈利
    cost   = f.get("cost_per_kg", 0)
    margin = f.get("profit_margin", 0)
    if cost > 0:
        profit_str = f"盈利{margin:.2f}元/公斤" if margin > 0 else f"亏损{abs(margin):.2f}元/公斤"
        lines.append(f"  养殖成本: {cost:.2f}元/公斤  当前{profit_str}")

    # 供给
    sow = f.get("sow_inventory", 0)
    sow_chg = f.get("sow_3m_change", 0)
    if sow > 0:
        lines.append(f"  能繁母猪: {sow:.2f}百万头  近3月: {sow_chg:+.2f}百万头 → {f.get('supply_signal','N/A')}")

    # 综合结论
    lines.append(f"  综合评分: {f.get('fundamental_score',0):+d}/3  {dir_cn}（{str_cn}）")
    lines.append(f"  判断: {v['summary']}")

    return "\n".join(lines)


def _empty_fundamentals() -> Dict[str, Any]:
    return {
        "spot_price": 0.0, "spot_7d_change": 0.0, "spot_30d_change": 0.0,
        "spot_trend": "FLAT", "national_avg": 0.0, "three_way_price": 0.0,
        "futures_price": 0.0, "basis_pct": 0.0, "basis_signal": "NORMAL",
        "cost_per_kg": 0.0, "profit_margin": 0.0, "profit_signal": "BREAKEVEN",
        "sow_inventory": 0.0, "sow_3m_change": 0.0, "supply_signal": "STABLE",
        "price_index": 0.0, "index_vs_ma4": 1.0, "index_7d_change": 0.0,
        "fundamental_score": 0, "fundamental_signals": [],
    }
