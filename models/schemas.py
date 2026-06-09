from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class KlineData(BaseModel):
    symbol: str
    interval: str
    timestamps: List[str]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[float]
    open_interests: Optional[List[float]] = None


class MarketSnapshot(BaseModel):
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume: float
    open_interest: float
    change_pct: float
    timestamp: str


class TechSignal(BaseModel):
    direction: Literal["LONG", "SHORT", "NEUTRAL"]
    strength: Literal["STRONG", "MODERATE", "WEAK"]
    reasoning: str


class TechReport(BaseModel):
    symbol: str
    signal: TechSignal
    key_support: float
    key_resistance: float
    stop_loss: float
    target_price: float
    confidence: float = Field(ge=0, le=1)
    indicators: Dict[str, Any] = {}


class FundReport(BaseModel):
    symbol: str
    net_flow: Literal["INFLOW", "OUTFLOW", "NEUTRAL"]
    basis_status: Literal["CONTANGO", "BACKWARDATION", "FLAT"]
    top_member_action: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class MacroReport(BaseModel):
    sector: str
    macro_trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    key_drivers: List[str]
    risk_events: List[str]
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class VisionReport(BaseModel):
    symbol: str
    chart_pattern: str
    visual_signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    pattern_completion_pct: float = Field(ge=0, le=100)
    reasoning: str


class RegimeOutput(BaseModel):
    regime: Literal["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "HIGH_VOLATILITY", "CRISIS"]
    adx_value: float
    vix_equivalent: float
    recommended_weights: Dict[str, float]
    reasoning: str


class ScenarioPath(BaseModel):
    probability: float = Field(ge=0, le=1)
    description: str
    target_price: float
    key_trigger: str


class ScenarioReport(BaseModel):
    paths: List[ScenarioPath]
    worst_case_loss_pct: float
    best_case_gain_pct: float = 0.0


class CausalEffect(BaseModel):
    direction: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    strength: Literal["STRONG", "MODERATE", "WEAK"]
    chain: str
    confidence: float = Field(ge=0, le=1)


class MemoryCase(BaseModel):
    date: str
    similarity: float = Field(ge=0, le=1)
    description: str
    subsequent_5d_return: float


class MemoryReport(BaseModel):
    similar_cases: List[MemoryCase]
    historical_win_rate: float = Field(ge=0, le=1)
    avg_profit_loss_ratio: float
    conclusion: str


class TrapType(BaseModel):
    type: Literal["BULL_TRAP", "BEAR_TRAP", "SHAKEOUT", "NONE"]
    current_phase: str
    trigger_to_confirm: str
    confidence: float = Field(ge=0, le=1)


class TrapAnalysisReport(BaseModel):
    symbol: str
    trap: TrapType
    reasoning: str


class ABMResult(BaseModel):
    liquidity_vacuums: List[Dict[str, Any]]
    projected_short_term_balance: Literal["BUYER_DOMINANT", "SELLER_DOMINANT", "BALANCED"]
    optimal_entry_offset: float


class CrowdingLevel(BaseModel):
    symbol: str
    score: int = Field(ge=0, le=100)
    warning: bool
    similar_funds_pct: float
    reasoning: str


class CommanderDecision(BaseModel):
    symbol: str
    action: Literal["LONG", "SHORT", "WAIT"]
    confidence: float = Field(ge=0, le=1)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    position_size_pct: float = Field(default=0.0, ge=0, le=1)
    reasoning: str
    posterior_probability: float = Field(ge=0, le=1)
    veto_reasons: List[str] = []


class Order(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["LIMIT", "MARKET", "STOP"]
    price: Optional[float] = None
    quantity: float
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"


class RiskOrder(BaseModel):
    decision: CommanderDecision
    orders: List[Order]
    max_loss: float
    risk_pct: float
    execution_notes: str


class MetaCognitionOutput(BaseModel):
    date: str
    reflection: str
    identified_biases: List[str] = []
    parameter_adjustment_suggestion: str
    modified_config: Dict[str, Any] = {}


class OnlineLearnerState(BaseModel):
    model_version: str
    recent_accuracy: float = Field(ge=0, le=1)
    drift_detected: bool
    drift_details: Optional[str] = None


class StrategyHypothesis(BaseModel):
    name: str
    logic_description: str
    pseudocode: str
    robustness_score: float = Field(ge=0, le=1)
    market_regime_fit: List[str]


class StrategyEvolverOutput(BaseModel):
    analysis_period: str
    new_hypotheses: List[StrategyHypothesis]
    deprecated_strategies: List[str] = []
    reasoning: str


class IgniterSignal(BaseModel):
    symbol: str
    triggered: bool
    trigger_reason: str
    trigger_price: float
    timestamp: str


class ScannerOutput(BaseModel):
    candidates: List[str]
    scan_timestamp: str
    total_screened: int
    selection_criteria: str
