# 先知期货认知交易系统 (Prophet Futures Cognitive Trading System)

基于 LangChain + LangGraph 构建的多智能体期货交易决策系统，集成贝叶斯融合、因果推理、历史记忆检索和多维情景分析。

---

## 系统架构

```
prophet_futures/
├── main.py                    # 入口文件（支持 --mode backtest/paper_trading）
├── config.yaml                # 系统配置
├── requirements.txt           # 依赖
├── agents/                    # 所有 Agent 模块
│   ├── scanner.py             # 市场扫描器
│   ├── technician.py          # 技术分析师（指标硬编码计算）
│   ├── fund_analyst.py        # 资金流向分析
│   ├── macro_analyst.py       # 宏观分析
│   ├── vision_tech.py         # 视觉图表识别（多模态）
│   ├── regime_detector.py     # 市场气象台（ADX/ATR规则）
│   ├── commander.py           # 综合决策官（贝叶斯融合）
│   ├── igniter.py             # 微观点火器
│   ├── risk_manager.py        # 风控与订单生成
│   ├── scenario_engine.py     # 多维情景规划
│   ├── causal_reasoner.py     # 因果推断
│   ├── memory_retriever.py    # 历史记忆检索
│   ├── trap_detector.py       # 主力陷阱识别
│   ├── crowding_radar.py      # 拥挤度雷达（规则计算）
│   ├── meta_cognition.py      # 元认知日度反思
│   ├── strategy_evolver.py    # 策略进化器（月度）
│   ├── online_learner.py      # 在线学习（River库）
│   ├── execution_rl.py        # RL最优执行
│   └── abm_simulator.py       # ABM沙盘接口
├── tools/                     # 工具函数（纯Python计算）
│   ├── indicators.py          # 所有技术指标（MA/MACD/RSI/ATR/ADX等）
│   ├── market_data.py         # 行情获取（akshare + 合成数据兜底）
│   ├── fund_data.py           # 资金持仓数据
│   ├── macro_data.py          # 宏观数据
│   ├── causal_graph.py        # 因果图（预定义有向图）
│   ├── memory_store.py        # 向量记忆库（ChromaDB）
│   ├── abm_engine.py          # ABM微观沙盘
│   ├── backtest.py            # 回测引擎
│   ├── rl_execution_env.py    # RL执行环境
│   └── llm_utils.py           # LLM客户端工具
├── prompts/                   # 所有Agent系统提示词（.md文件）
├── models/
│   └── schemas.py             # 所有 Pydantic 数据模型
├── graph/
│   ├── state.py               # LangGraph 工作流状态
│   └── workflow.py            # 完整工作流（含并行节点+条件边）
└── utils/
    ├── logger.py              # 日志配置
    └── portfolio_analytics.py # 组合统计分析
```

---

## 安装

### 1. 创建虚拟环境

```bash
cd prophet_futures
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 方法一：环境变量（推荐）
export ANTHROPIC_API_KEY="sk-ant-..."

# 方法二：.env 文件
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

如需使用 OpenAI，修改 `config.yaml`：
```yaml
system:
  llm_provider: openai
  llm_model: gpt-4o
```
并设置 `OPENAI_API_KEY`。

---

## 运行

### 回测模式（指定日期）

```bash
python main.py --mode backtest --date 2024-06-01
```

### 模拟交易模式

```bash
python main.py --mode paper_trading
```

### 指定分析品种

```bash
python main.py --mode paper_trading --symbol cu
```

### 调整日志级别

```bash
python main.py --mode paper_trading --log-level DEBUG
```

---

## 工作流说明

```
START
  └─ Scanner（扫描全市场）
       └─ AnalyzeSymbol（并行：技术+资金+宏观+视觉）
              └─ AdvancedCognition（并行：情景+因果+记忆+陷阱+拥挤度）
                    └─ Commander（贝叶斯融合 + 5项否决规则）
                           ├─ [action=WAIT] → MetaCognition → END
                           └─ [action=LONG/SHORT] → Igniter
                                  ├─ [未触发] → MetaCognition → END
                                  └─ [已触发] → RiskManager → MetaCognition → END
```

### Commander 否决规则（任一触发则 WAIT）
1. 拥挤度评分 > 85
2. 历史胜率 < 45%
3. 最坏情景亏损 > 5%
4. 因果引擎返回 NEGATIVE+STRONG（政策利空）
5. 陷阱检测置信度 > 70% 且方向与计划一致

### 指标计算原则
所有技术指标（MA、MACD、RSI、ATR、ADX、布林带）均在 `tools/indicators.py` 中用 Python/NumPy **硬编码计算**，绝不允许 LLM 生成数值。

---

## 配置说明

主要配置项（`config.yaml`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `system.llm_provider` | LLM提供商 (anthropic/openai) | anthropic |
| `system.llm_model` | 模型名称 | claude-sonnet-4-6 |
| `risk.capital` | 账户资金（元） | 1,000,000 |
| `risk.max_single_risk_pct` | 单品种最大风险比例 | 2% |
| `risk.daily_drawdown_halt_pct` | 日内熔断线 | 3% |
| `advanced.crowding.warning_threshold` | 拥挤度预警阈值 | 80 |
| `advanced.memory.db_path` | 向量库存储路径 | ./vector_db |

---

## 扩展与定制

### 添加新品种
在 `config.yaml` 的 `markets.futures` 列表中添加品种代码，并在 `tools/causal_graph.py` 的 `CAUSAL_EDGES` 中补充该品种的因果关系。

### 自定义提示词
直接编辑 `prompts/` 目录下对应 Agent 的 `.md` 文件，无需修改 Python 代码。

### 接入实盘
在 `tools/market_data.py` 中替换 `get_realtime_quote` 和 `get_tick_data` 的实现，对接券商 API。风控约束在 `agents/risk_manager.py` 中统一管理。

---

## 依赖说明

| 依赖 | 用途 | 必须 |
|------|------|------|
| langchain / langgraph | 工作流编排 | ✅ |
| langchain-anthropic | Anthropic LLM | ✅（或openai） |
| akshare | 期货行情数据 | ✅ |
| pandas / numpy | 数据处理 | ✅ |
| pydantic>=2 | 输出校验 | ✅ |
| chromadb | 向量记忆库 | 可选 |
| matplotlib | K线图表（视觉Agent） | 可选 |
| river | 在线学习/漂移检测 | 可选 |
| gymnasium | RL执行环境 | 可选 |
| networkx | 因果图 | 可选 |

所有可选依赖缺失时，系统自动切换到规则/启发式兜底逻辑，不影响主流程运行。

---

## 免责声明

本系统仅供技术学习和研究使用，不构成任何投资建议。期货交易存在较大风险，请在充分理解风险的前提下进行投资决策。
