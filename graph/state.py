"""LangGraph state definition for the trading workflow."""

from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from models.schemas import (
    ScannerOutput, TechReport, FundReport, MacroReport, VisionReport,
    RegimeOutput, ScenarioReport, CausalEffect, MemoryReport,
    TrapAnalysisReport, CrowdingLevel, CommanderDecision,
    IgniterSignal, RiskOrder, MetaCognitionOutput, OnlineLearnerState,
)


class TradingState(TypedDict, total=False):
    # Scan
    candidates:        List[str]
    current_symbol:    str
    active_symbol:     Optional[str]
    scan_output:       Optional[ScannerOutput]
    # Analysis
    technical_report:  Optional[TechReport]
    fund_report:       Optional[FundReport]
    macro_report:      Optional[MacroReport]
    vision_report:     Optional[VisionReport]
    regime:            Optional[RegimeOutput]
    # Advanced cognition
    scenario_report:   Optional[ScenarioReport]
    causal_report:     Optional[CausalEffect]
    memory_report:     Optional[MemoryReport]
    trap_report:       Optional[TrapAnalysisReport]
    crowding:          Optional[CrowdingLevel]
    # Decision & execution
    commander_decision:Optional[CommanderDecision]
    igniter_trigger:   Optional[IgniterSignal]
    risk_order:        Optional[RiskOrder]
    # Post-trade
    meta_reflection:   Optional[MetaCognitionOutput]
    learner_state:     Optional[OnlineLearnerState]
    # Meta
    final_output:      str
    daily_summary:     Dict[str, Any]
    mode:              str
    date:              Optional[str]
    errors:            List[str]
