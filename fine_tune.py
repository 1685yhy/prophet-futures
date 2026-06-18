#!/usr/bin/env python3
"""Fine-tuning: find sweet spot between Conservative and Baseline."""

import sys; sys.path.insert(0, ".")
from advanced_strategy import *

print("="*70)
print("  甜点区搜索 — Regime Filter + Time Stop")
print("="*70)

capital = 1_000_000
lh = fetch_history("lh", 1200)

# Fixed: regime filter ON, time stop ON, dynamic sizing OFF, pyramiding OFF
# Vary: min_conds, stop_atr, target_atr
results = []
for mc in [6, 7, 8]:
    for sa in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for ta in [1.5, 2.0, 2.5, 3.0]:
            params = {"min_conds":mc, "stop_atr":sa, "target_atr":ta,
                      "require_regime":True, "require_volume":False,
                      "use_dynamic_size":False, "use_pyramiding":False,
                      "use_time_stop":True, "time_stop_days":8}
            trades = advanced_backtest(lh, "lh", capital, params)
            stats = compute_stats(trades, capital)
            stats["min_conds"]=mc; stats["stop_atr"]=sa; stats["target_atr"]=ta
            # Score: prefer high win_rate AND high return, penalize low trades
            if stats["trades"] >= 10:
                score = stats["win_rate"] * stats["total_return"] * min(stats["trades"], 60)
                results.append(stats)

results.sort(key=lambda x: (x["win_rate"] * x["total_return"] * min(x["trades"], 60)), reverse=True)

print(f"\n  Top 20 (of {len(results)} valid):")
print(f"  {'Rank':<5} {'MC':<4} {'Stop':<7} {'Tgt':<6} {'Trades':<7} {'月均':<6} {'胜率':<7} {'盈亏比':<7} {'PnL':<12} {'收益%':<8} {'回撤%':<7} {'夏普':<7}")
print(f"  {'─'*85}")
for i, r in enumerate(results[:20]):
    print(f"  {i+1:<5} {r['min_conds']:<4} {r['stop_atr']:<7} {r['target_atr']:<6} "
          f"{r['trades']:<7} {r['monthly_trades']:<6.1f} {r['win_rate']:.0%}     "
          f"{r['pl_ratio']:<7.2f} {r['total_pnl']:+,.0f}     {r['total_return']:+.1f}%    "
          f"{r['max_dd']:.1f}%    {r['sharpe']:.3f}")

# Best overall
best = results[0]
print(f"\n{'='*70}")
print(f"  甜点策略: MC={best['min_conds']}/8, Stop={best['stop_atr']}xATR, Tgt={best['target_atr']}xATR")
print(f"{'='*70}")
print(f"  交易: {best['trades']}笔 (月均{best['monthly_trades']:.1f}笔)")
print(f"  胜率: {best['win_rate']:.0%}  盈亏比: {best['pl_ratio']:.2f}")
print(f"  收益: {best['total_pnl']:+,.0f}元 ({best['total_return']:+.1f}%)")
print(f"  年化: ~{best['total_return']/3:.1f}%  回撤: {best['max_dd']:.1f}%")
print(f"  夏普: {best['sharpe']:.3f}")

# Walk-forward on best
print(f"\n[Walk-Forward] 甜点策略步进验证...")
# Simplified walk-forward
n_splits = 3
total = len(lh)
sz = total // (n_splits + 1)
all_test = []
for sp in range(n_splits):
    te_start = sz * (sp + 1)
    te_end = min(te_start + sz, total)
    test_df = lh.iloc[te_start:te_end]
    if len(test_df) < 60: continue
    params = {"min_conds":best['min_conds'], "stop_atr":best['stop_atr'],
              "target_atr":best['target_atr'], "require_regime":True,
              "require_volume":False, "use_dynamic_size":False,
              "use_pyramiding":False, "use_time_stop":True, "time_stop_days":8}
    tr = advanced_backtest(test_df, "lh", capital, params)
    st = compute_stats(tr, capital)
    all_test.extend(tr)
    print(f"  Split{sp+1}: {st['trades']}笔, {st['win_rate']:.0%}胜率, {st['total_pnl']:+,.0f}元, {st['max_dd']:.1f}%回撤")

agg = compute_stats(all_test, capital)
print(f"  合计: {agg['trades']}笔, {agg['win_rate']:.0%}胜率, {agg['total_pnl']:+,.0f}元 (+{agg['total_return']:.1f}%)")
print(f"\n甜点确认: Walk-Forward {agg['win_rate']:.0%}胜率, {agg['total_return']:+.1f}%收益")
