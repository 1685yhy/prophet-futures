"""LLM client utilities — provider-agnostic wrapper with correct ReAct prompt construction."""

import logging
import yaml
from pathlib import Path
from typing import Any, Optional, Type, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_config_cache: Optional[dict] = None
_llm_available: Optional[bool] = None   # None=未检测, True=可用, False=不可用


def load_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            _config_cache = yaml.safe_load(f)
    else:
        _config_cache = {
            "system": {"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-6"},
            "risk": {"capital": 1000000},
        }
    return _config_cache


def get_llm(temperature: float = 0.1, model_override: Optional[str] = None) -> Any:
    """Build and return an LLM client based on config."""
    cfg      = load_config()
    provider = cfg.get("system", {}).get("llm_provider", "anthropic")
    model    = model_override or cfg.get("system", {}).get("llm_model", "claude-sonnet-4-6")

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096)
        except ImportError:
            logger.warning("langchain_anthropic not installed; trying openai fallback")

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature)
        except ImportError:
            raise RuntimeError("Install langchain_anthropic or langchain_openai")

    raise ValueError(f"Unknown LLM provider: {provider}")


def check_llm_connectivity() -> bool:
    """
    探针：尝试用最小请求验证 LLM 是否可达。
    结果缓存到 _llm_available，避免重复等待超时。
    """
    global _llm_available
    if _llm_available is not None:
        return _llm_available
    try:
        llm = get_llm(temperature=0.0)
        llm.invoke("hi")
        _llm_available = True
        logger.info("LLM connectivity: OK")
    except Exception as e:
        _llm_available = False
        logger.warning("LLM connectivity: UNAVAILABLE (%s) — all agents will use fallback rules", e)
    return _llm_available


def is_llm_available() -> bool:
    """快速检查，不触发探针（仅返回缓存状态）。"""
    return _llm_available is True


def load_prompt(agent_name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{agent_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    logger.warning("Prompt file not found: %s", prompt_path)
    return f"You are a {agent_name} agent in a futures trading system. Provide structured analysis."


def build_react_prompt(system_prompt: str):
    """Build a ChatPromptTemplate compatible with create_react_agent."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nTools available:\n{tools}\nTool names: {tool_names}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def build_agent_executor(
    agent_name: str,
    tools: List[Any],
    temperature: float = 0.1,
    max_iterations: int = 5,
    verbose: bool = False,
) -> Any:
    """Build a fully wired AgentExecutor using the correct ChatPromptTemplate."""
    try:
        from langchain.agents import create_react_agent, AgentExecutor
    except ImportError:
        raise RuntimeError("langchain not installed: pip install langchain")

    llm    = get_llm(temperature=temperature)
    prompt = build_react_prompt(load_prompt(agent_name))
    try:
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    except TypeError:
        agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent, tools=tools, verbose=verbose,
        max_iterations=max_iterations, handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def invoke_structured(
    agent_name: str,
    tools: List[Any],
    input_text: str,
    schema: Type[BaseModel],
    temperature: float = 0.1,
    max_iterations: int = 5,
) -> Optional[BaseModel]:
    """
    Two-phase: ReAct agent gathers data with tools, then structured-output LLM
    converts gathered text to validated Pydantic schema.
    Returns None on any failure — caller applies heuristic fallback.

    Short-circuit: if LLM is known unavailable (_llm_available=False),
    skip immediately without waiting for timeout.
    """
    if _llm_available is False:
        logger.debug("[%s] LLM unavailable, using fallback", agent_name)
        return None

    system_prompt = load_prompt(agent_name)

    # Phase 1 — gather data with tools
    try:
        executor      = build_agent_executor(agent_name, tools, temperature, max_iterations)
        result        = executor.invoke({"input": input_text})
        gathered_text = result.get("output", "")
        if not gathered_text:
            raise ValueError("empty output from ReAct agent")
    except Exception as e:
        logger.warning("[%s] ReAct phase failed (%s) — skipping Phase 2", agent_name, e)
        return None  # Phase 1 失败直接返回，不浪费 token

    # Phase 2 — force structured output
    try:
        llm            = get_llm(temperature=temperature)
        structured_llm = llm.with_structured_output(schema)
        return structured_llm.invoke(
            f"{system_prompt}\n\n"
            f"Based on the following gathered data, produce a valid {schema.__name__}:\n\n"
            f"{gathered_text}"
        )
    except Exception as e:
        logger.warning("[%s] Structured output phase failed: %s", agent_name, e)
        return None


def parse_json_output(text: str) -> dict:
    """Fallback JSON extractor for agents that can't use with_structured_output."""
    import json, re
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group())
            except json.JSONDecodeError:
                continue
    logger.warning("Could not parse JSON from LLM output: %s", text[:200])
    return {}
