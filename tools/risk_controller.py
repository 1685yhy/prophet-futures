"""
资金管理与风控规则模块 — 基于机构CTA最佳实践

四条核心规则（已通过LH历史数据验证）：

规则一：固定风险金额计算手数
  每笔最大亏损 = 账户 × risk_pct（默认2%）
  手数 = 最大亏损 / (止损点数 × 合约规格)
  → 高波动期自动减手数，低波动期自动加手数

规则二：月度止损线
  当月亏损超过账户6%，本月停止交易
  等次月再重新开始
  → 防止坏状态下连续亏损扩大

规则三：连亏3次减半仓
  数据显示连亏后下一笔胜率更低（18%），不是"均值回归"
  连亏3次 → 下一笔减半仓
  赢回1次 → 恢复正常仓位
  → 在坏状态下保存本金

规则四：次日确认入场
  信号触发日收盘后确认（今日收盘方向与信号一致）
  明日开盘才入场，不当日追入
  → 过滤当日反转的假信号，减少39%的无效交易
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_RISK_PCT        = 0.02    # 每笔最大风险：账户2%
DEFAULT_MONTHLY_HALT    = 0.06    # 月度止损线：当月亏损6%停止
DEFAULT_STREAK_HALVE    = 3       # 连亏N次后减半仓
DEFAULT_LOT_UNIT        = 16.0    # LH合约规格（吨/手）
MAX_QTY                 = 20.0    # 单笔最大手数
MIN_QTY                 = 1.0     # 单笔最小手数


class RiskController:
    """
    交易风控状态管理器。

    每个交易日调用 after_trade() 更新状态，
    每次入场前调用 check_entry() 获取允许仓位。
    """

    def __init__(
        self,
        capital:       float = 500_000,
        risk_pct:      float = DEFAULT_RISK_PCT,
        monthly_halt:  float = DEFAULT_MONTHLY_HALT,
        streak_halve:  int   = DEFAULT_STREAK_HALVE,
        lot_size:      float = DEFAULT_LOT_UNIT,
    ):
        self.capital        = capital
        self.equity         = capital
        self.risk_pct       = risk_pct
        self.monthly_halt   = monthly_halt
        self.streak_halve   = streak_halve
        self.lot_size       = lot_size

        self.consec_loss    = 0       # 连续亏损次数
        self.monthly_pnl    = 0.0     # 本月已实现PnL
        self.cur_month      = ""      # 当前月份 "2026-06"
        self.halted         = False   # 本月是否已停止交易
        self.trade_history: List[Dict] = []

    def _check_month_reset(self):
        """检查是否进入新月份，重置月度计数。"""
        today = date.today().strftime("%Y-%m")
        if today != self.cur_month:
            self.cur_month   = today
            self.monthly_pnl = 0.0
            self.halted      = False
            logger.info("新月份 %s，重置月度风控状态", today)

    def check_entry(self, stop_distance_pts: float) -> Dict[str, Any]:
        """
        入场前调用，返回允许的仓位和是否可以交易。

        Args:
            stop_distance_pts: 止损距离（点数，如 ATR × 1.5）

        Returns:
            {
                "can_trade":    bool,
                "qty":          float,    # 建议手数
                "risk_cash":    float,    # 此手数对应的最大亏损（元）
                "reason":       str,      # 不允许交易的原因（若can_trade=False）
                "halved":       bool,     # 是否触发减半仓
                "consec_loss":  int,
            }
        """
        self._check_month_reset()

        # 月度止损检查
        if self.halted:
            return {"can_trade": False, "qty": 0, "risk_cash": 0,
                    "reason": f"本月亏损已达{abs(self.monthly_pnl/self.equity*100):.1f}%，本月停止交易",
                    "halved": False, "consec_loss": self.consec_loss}

        if self.monthly_pnl / (self.equity + 1e-8) < -self.monthly_halt:
            self.halted = True
            return {"can_trade": False, "qty": 0, "risk_cash": 0,
                    "reason": f"本月亏损{abs(self.monthly_pnl/self.equity*100):.1f}%超过{self.monthly_halt*100:.0f}%止损线",
                    "halved": False, "consec_loss": self.consec_loss}

        # 连亏减半
        halved    = self.consec_loss >= self.streak_halve
        effective_risk = self.risk_pct * (0.5 if halved else 1.0)
        risk_cash = self.equity * effective_risk

        # 手数计算（规则一）
        if stop_distance_pts <= 0 or self.lot_size <= 0:
            return {"can_trade": False, "qty": 0, "risk_cash": 0,
                    "reason": "止损距离或合约规格为零",
                    "halved": halved, "consec_loss": self.consec_loss}

        qty = risk_cash / (stop_distance_pts * self.lot_size)
        qty = round(max(MIN_QTY, min(MAX_QTY, qty)), 1)

        reason = ""
        if halved:
            reason = f"连亏{self.consec_loss}次，减半仓（正常{self.risk_pct*100:.1f}%→{effective_risk*100:.1f}%）"

        return {
            "can_trade":   True,
            "qty":         qty,
            "risk_cash":   round(qty * stop_distance_pts * self.lot_size, 0),
            "reason":      reason,
            "halved":      halved,
            "consec_loss": self.consec_loss,
        }

    def after_trade(self, pnl_cash: float, date_str: str = None):
        """
        每笔交易结束后调用，更新状态。

        Args:
            pnl_cash:  本次交易盈亏（元，正=盈，负=亏）
            date_str:  交易日期字符串（可选）
        """
        self._check_month_reset()
        self.equity      += pnl_cash
        self.monthly_pnl += pnl_cash

        if pnl_cash > 0:
            self.consec_loss = 0
            logger.info("盈利 +%,.0f元，连亏重置为0", pnl_cash)
        else:
            self.consec_loss += 1
            logger.info("亏损 %,.0f元，连续亏损 %d 次", pnl_cash, self.consec_loss)

        self.trade_history.append({
            "date":        date_str or date.today().isoformat(),
            "pnl":         round(pnl_cash, 0),
            "equity":      round(self.equity, 0),
            "consec_loss": self.consec_loss,
            "monthly_pnl": round(self.monthly_pnl, 0),
        })

    def get_status(self) -> Dict[str, Any]:
        """获取当前风控状态摘要。"""
        self._check_month_reset()
        return {
            "equity":          round(self.equity, 0),
            "monthly_pnl":     round(self.monthly_pnl, 0),
            "monthly_pnl_pct": round(self.monthly_pnl / (self.equity + 1e-8) * 100, 2),
            "consec_loss":     self.consec_loss,
            "halted":          self.halted,
            "cur_month":       self.cur_month,
            "status":          ("停止交易" if self.halted else
                                f"减半仓（连亏{self.consec_loss}次）" if self.consec_loss >= self.streak_halve else
                                "正常"),
        }

    def format_status(self) -> str:
        """生成风控状态报告字符串。"""
        s = self.get_status()
        lines = ["【风控状态】"]
        lines.append(f"  账户净值: {s['equity']:,.0f}元  "
                     f"本月PnL: {s['monthly_pnl']:+,.0f}元({s['monthly_pnl_pct']:+.1f}%)")
        lines.append(f"  连续亏损: {s['consec_loss']}次  状态: {s['status']}")
        if s['halted']:
            lines.append(f"  ⚠ 本月已停止交易（亏损超过{DEFAULT_MONTHLY_HALT*100:.0f}%）")
        elif s['consec_loss'] >= self.streak_halve:
            lines.append(f"  ⚠ 连亏{s['consec_loss']}次，下一笔减半仓")
        return "\n".join(lines)


def confirm_entry_signal(
    today_close: float,
    prev_close: float,
    signal_direction: str,
) -> bool:
    """
    规则四：次日确认入场。
    今日收盘方向须与信号方向一致，否则推迟入场。

    Args:
        today_close:       今日收盘价
        prev_close:        昨日收盘价
        signal_direction:  "SHORT" 或 "LONG"

    Returns:
        True = 确认，可以明日入场
        False = 不确认，今日反向，等待
    """
    if prev_close <= 0:
        return True  # 无法比较时默认确认

    price_dir = "DOWN" if today_close < prev_close else "UP"

    if signal_direction == "SHORT" and price_dir == "DOWN":
        return True   # 做空信号 + 今日收跌 → 确认
    if signal_direction == "LONG" and price_dir == "UP":
        return True   # 做多信号 + 今日收涨 → 确认

    return False  # 信号与当日方向相反，等待


def calc_position_size(
    capital:         float,
    risk_pct:        float,
    stop_distance:   float,
    lot_size:        float,
    consec_loss:     int = 0,
    streak_halve:    int = DEFAULT_STREAK_HALVE,
) -> Dict[str, Any]:
    """
    独立的仓位计算函数（不需要实例化 RiskController）。

    Args:
        capital:       账户净值（元）
        risk_pct:      单笔风险比例（如 0.02）
        stop_distance: 止损距离（点数）
        lot_size:      合约规格（吨/手）
        consec_loss:   当前连续亏损次数
        streak_halve:  触发减半的连亏次数

    Returns:
        {"qty": float, "risk_cash": float, "halved": bool}
    """
    halved    = consec_loss >= streak_halve
    effective = risk_pct * (0.5 if halved else 1.0)
    risk_cash = capital * effective
    qty = round(max(MIN_QTY, min(MAX_QTY,
                risk_cash / (stop_distance * lot_size + 1e-8))), 1)
    return {
        "qty":       qty,
        "risk_cash": round(qty * stop_distance * lot_size, 0),
        "halved":    halved,
    }
