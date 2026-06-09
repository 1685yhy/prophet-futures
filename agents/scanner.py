"""Scanner Agent — screens the full futures universe for trading candidates."""

import logging
from datetime import datetime
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.market_data import get_contracts, get_realtime_quote
from tools.fund_data import get_volume_oi
from tools.macro_data import get_market_index
from models.schemas import ScannerOutput

logger = logging.getLogger(__name__)


def run_scanner() -> ScannerOutput:
    result = invoke_structured(
        agent_name="scanner",
        tools=[
            Tool(name="get_contracts",
                 func=lambda _: str(get_contracts()),
                 description="Get list of active futures contract symbols"),
            Tool(name="get_volume_oi",
                 func=lambda sym: str(get_volume_oi(sym.strip())),
                 description="Get volume and open interest for a symbol. Input: symbol"),
            Tool(name="get_realtime_quote",
                 func=lambda sym: str(get_realtime_quote(sym.strip()).model_dump()),
                 description="Get realtime quote for a symbol. Input: symbol"),
            Tool(name="get_market_index",
                 func=lambda _: str(get_market_index()),
                 description="Get major market index data"),
        ],
        input_text="Scan all active futures contracts and select the top 3-8 trading candidates.",
        schema=ScannerOutput, temperature=0.1, max_iterations=5,
    )
    if result is not None:
        return result

    logger.warning("Scanner fallback")
    contracts = get_contracts()[:5]
    return ScannerOutput(
        candidates=contracts,
        scan_timestamp=datetime.now().isoformat(),
        total_screened=len(get_contracts()),
        selection_criteria="Fallback: top active contracts by default ordering",
    )
