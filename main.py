#!/usr/bin/env python3
"""
Prophet Futures Cognitive Trading System — Entry Point

Usage:
  python main.py --mode paper_trading [--symbol lh]
  python main.py --mode backtest --date 2025-06-01 [--backtest-days 180] [--symbols rb,lh,sc]
  python main.py --mode build_memory [--symbols rb,i,lh,jd,sc] [--start 20230101]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging
from tools.llm_utils import load_config, check_llm_connectivity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prophet Futures Cognitive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper_trading", "build_memory"],
        default="paper_trading",
        help="Operating mode",
    )
    parser.add_argument("--date",   default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=None, help="Override symbol (paper_trading)")
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols for backtest/build_memory (e.g. rb,lh,sc)",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=180,
        help="Number of days for backtest window (default: 180)",
    )
    parser.add_argument(
        "--start", default="20230101",
        help="Start date for build_memory mode (YYYYMMDD, default: 20230101)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args()


def run_paper_trading(symbol_override=None):
    logger = logging.getLogger(__name__)
    logger.info("Starting PAPER TRADING mode")

    from graph.workflow import get_compiled_workflow
    from graph.state import TradingState
    from tools.memory_store import init_vector_db

    cfg = load_config()
    init_vector_db(cfg.get("advanced", {}).get("memory", {}).get("db_path", "./vector_db"))

    workflow = get_compiled_workflow()
    initial: TradingState = {
        "mode":          "paper_trading",
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "candidates":    [symbol_override] if symbol_override else [],
        "errors":        [],
        "daily_summary": {},
        "final_output":  "",
    }

    final = workflow.invoke(initial)
    print("\n" + final.get("final_output", "No output generated"))

    if final.get("risk_order") and final["risk_order"].orders:
        o = final["risk_order"]
        print(f"\n订单: {len(o.orders)} 笔  最大亏损: {o.max_loss:.2f}  风险: {o.risk_pct:.2%}")
        for i, order in enumerate(o.orders, 1):
            print(f"  [{i}] {order.side} {order.quantity:.1f}手 {order.symbol} "
                  f"@ {order.price or 'MARKET'}")
    else:
        print("\n无交易信号 (WAIT 或执行未触发)")

    return final


def run_backtest(date, symbols, backtest_days):
    logger = logging.getLogger(__name__)
    logger.info("Starting BACKTEST mode: %s, days=%d, symbols=%s", date, backtest_days, symbols)

    from tools.backtest import run_backtest as bt
    from utils.portfolio_analytics import generate_trade_report

    result = bt(date=date, symbols=symbols, backtest_days=backtest_days)

    if "error" in result:
        print(f"回测失败: {result['error']}")
        return result

    print(generate_trade_report(result.get("trades", [])))
    print(f"\n回测区间: 截至 {result['date_range']}, {backtest_days} 天")
    print(f"品种: {', '.join(result['symbols'])}")
    print(f"总交易: {result['total_trades']} 笔")
    print(f"胜率:   {result['win_rate']:.1%}")
    print(f"盈亏比: {result['pl_ratio']:.2f}")
    print(f"夏普:   {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown_pct']:.2f}%")
    print(f"总收益: {result['total_pnl']:+,.2f} 元 ({result['total_return_pct']:+.2f}%)")

    if result.get("trades"):
        print(f"\n最近5笔交易:")
        for t in result["trades"][-5:]:
            print(f"  {t['symbol']} {t['direction']:5s} {t['entry_date']}→{t['exit_date']} "
                  f"PnL={t['pnl']:+.1f} ({t['reason']})")

    return result


def run_build_memory(symbols, start_date):
    logger = logging.getLogger(__name__)
    logger.info("Building historical memory: symbols=%s, start=%s", symbols, start_date)

    from tools.history_builder import build_historical_memory, get_memory_stats
    from tools.llm_utils import load_config

    cfg     = load_config()
    db_path = cfg.get("advanced", {}).get("memory", {}).get("db_path", "./vector_db")

    print(f"构建历史记忆库...")
    print(f"品种: {symbols}")
    print(f"起始: {start_date}  存储: {db_path}")
    print()

    count = build_historical_memory(
        symbols=symbols,
        start_date=start_date,
        end_date=datetime.now().strftime("%Y%m%d"),
        db_path=db_path,
    )

    stats = get_memory_stats(db_path)
    print(f"\n完成！写入 {count} 条新记录")
    print(f"记忆库总量: {stats['total_records']} 条")
    if stats.get("symbols"):
        print(f"已覆盖品种: {', '.join(stats['symbols'])}")
    return {"written": count, "stats": stats}


def main():
    args   = parse_args()
    setup_logging(level=args.log_level, log_to_file=not args.no_log_file)
    logger = logging.getLogger(__name__)

    cfg = load_config()
    logger.info(
        "Prophet Futures starting | mode=%s | provider=%s | model=%s",
        args.mode,
        cfg.get("system", {}).get("llm_provider"),
        cfg.get("system", {}).get("llm_model"),
    )

    # LLM 连通性检查（paper_trading 模式才探针，其他模式不需要）
    if args.mode == "paper_trading":
        available = check_llm_connectivity()
        if available:
            print("LLM: 可用 ✓ — 将使用 AI 分析")
        else:
            print("LLM: 不可用 — 将使用规则 fallback（设置 ANTHROPIC_API_KEY 后可启用 AI）")

    # 解析品种列表
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = cfg.get("markets", {}).get("futures", ["rb", "i", "sc", "lh", "jd"])[:5]

    # 执行
    if args.mode == "paper_trading":
        result = run_paper_trading(symbol_override=args.symbol)
        sys.exit(0 if "error" not in result else 1)

    elif args.mode == "backtest":
        date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = run_backtest(date=date, symbols=symbols, backtest_days=args.backtest_days)
        sys.exit(0 if "error" not in result else 1)

    elif args.mode == "build_memory":
        result = run_build_memory(symbols=symbols, start_date=args.start)
        sys.exit(0)


if __name__ == "__main__":
    main()
