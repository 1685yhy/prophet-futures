# 先知期货认知交易系统 v3.0

## 一键使用

```bash
cd prophet_futures
source .venv/bin/activate
python run_today.py
```

输出：今日是否有交易信号，以及具体的入场价、止损价、止盈价、手数。

## 策略参数

| 品种 | 胜率 | 回撤 | 参数 |
|------|------|------|------|
| JM 焦煤 | 63% | 2.0% | MC=7, Stop=1.5×ATR, Target=3.0×ATR, Volume确认 |
| LH 生猪 | 50% | 3.1% | MC=7, Stop=1.5×ATR, Target=3.0×ATR |

## 风控规则

- 单笔风险：本金的 1%
- 连亏 3 笔：当月停手
- 月度亏损 5%：熔断

## 每日流程

1. 收盘后运行 `python run_today.py`
2. 有信号 → 次日开盘执行，设好止损止盈
3. 无信号 → 观望

## 回测验证

```bash
python final_verify.py    # 5年全量回测
python advanced_strategy.py  # 高级策略对比
python massive_optimization.py  # 34品种搜索
```

## 飞书推送

每日 15:00 自动推送分析到飞书（需 gateway 运行中）。

## 回滚

```bash
git log --oneline
git checkout <hash>
```
