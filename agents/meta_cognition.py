"""Meta-Cognition Agent — end-of-day reflection and parameter adjustment."""

import logging
import json
from datetime import datetime
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from models.schemas import MetaCognitionOutput

logger = logging.getLogger(__name__)


def run_meta_cognition(daily_summary: dict) -> MetaCognitionOutput:
    result = invoke_structured(
        agent_name="meta_cognition",
        tools=[Tool(name="get_daily_summary",
                    func=lambda _: json.dumps(daily_summary),
                    description="Get today's trading summary and decisions")],
        input_text="Reflect on today's trading performance. Call get_daily_summary to review decisions.",
        schema=MetaCognitionOutput, temperature=0.3, max_iterations=3,
    )
    if result is not None:
        return result

    logger.warning("Meta-cognition fallback")
    return MetaCognitionOutput(
        date=datetime.now().strftime("%Y-%m-%d"),
        reflection="System operated within expected parameters.",
        identified_biases=[],
        parameter_adjustment_suggestion="Monitor performance for 5 more trading days before adjustments",
        modified_config={},
    )
