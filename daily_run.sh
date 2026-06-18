#!/bin/bash
# Prophet Futures — Daily Auto Analysis
# Run via cron: 0 15 * * 1-5 /home/a/prophet_futures/prophet_futures/daily_run.sh

cd /home/a/prophet_futures/prophet_futures
source .venv/bin/activate
export DEEPSEEK_API_KEY=$(grep DEEPSEEK_API_KEY /home/a/.hermes/.env | cut -d= -f2)

echo "=== Prophet Futures Daily $(date +%Y-%m-%d) ==="

# 1. Full AI analysis
echo "[AI Analysis]"
python main.py --mode paper_trading --symbol lh --log-level WARNING 2>&1

echo ""
echo "[Daily Update]"
python main.py --mode daily_update --symbol lh --log-level WARNING 2>&1

echo ""
echo "[Backtest Check]"
python main.py --mode backtest --date $(date +%Y-%m-%d) --symbols lh --backtest-days 30 --log-level WARNING 2>&1

echo ""
echo "=== Done $(date) ==="
