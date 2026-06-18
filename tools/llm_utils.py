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

    if provider == "deepseek":
        import os
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=4096,
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            )
        except ImportError:
            raise RuntimeError("Install langchain-openai for DeepSeek support")

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
    import re
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Escape {品种} and other Chinese placeholders that aren't template variables.
    # The only variables ChatPromptTemplate should see are {tools}, {tool_names}, {input}.
    escaped = re.sub(r'\{(?!tools|tool_names|input\})([^}]*)\}', r'{{\1}}', system_prompt)

    return ChatPromptTemplate.from_messages([
        ("system", escaped + "\n\nTools available:\n{tools}\nTool names: {tool_names}"),
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
    """Build a tool-calling agent with structured output support."""
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = get_llm(temperature=temperature)
    system_prompt = load_prompt(agent_name)

    # Escape {品种} etc. so ChatPromptTemplate doesn't treat them as variables
    import re
    system_prompt = re.sub(r'\{(?!agent_scratchpad|input)([^}]*)\}', r'{{\1}}', system_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
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
        llm = get_llm(temperature=temperature)
        # DeepSeek doesn't support native structured output; use prompt-based JSON
        import json as _json
        schema_json = _json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        json_prompt = (
            f"{system_prompt}\n\n"
            f"Based on the following gathered data, output ONLY a valid JSON object "
            f"matching this schema (no markdown, no extra text):\n\n"
            f"Schema:\n{schema_json}\n\n"
            f"Gathered data:\n{gathered_text}"
        )
        raw = llm.invoke(json_prompt)
        raw_text = raw.content if hasattr(raw, 'content') else str(raw)
        parsed = parse_json_output(raw_text)
        if parsed and len(parsed) > 0:
            try:
                return schema(**parsed)
            except Exception as ve:
                # Try with defaults for common missing fields
                for field_name, field_info in schema.model_fields.items():
                    if field_name not in parsed:
                        default = field_info.default
                        if default is not None:
                            parsed[field_name] = default
                        elif field_info.annotation == str:
                            parsed[field_name] = ""
                        elif field_info.annotation == float:
                            parsed[field_name] = 0.0
                        elif field_info.annotation == int:
                            parsed[field_name] = 0
                        elif field_info.annotation == list:
                            parsed[field_name] = []
                try:
                    return schema(**parsed)
                except Exception as ve2:
                    logger.warning("[%s] Schema validation failed: %s — raw keys: %s",
                                 agent_name, ve2, list(parsed.keys())[:10])
                    return None
        logger.warning("[%s] Phase 2: empty parse from LLM output (len=%d): %.200s",
                      agent_name, len(raw_text), raw_text)
        return None
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
    # Try extracting from markdown code blocks
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    # Try finding any valid JSON object — use balanced brace matching
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
    logger.warning("Could not parse JSON from LLM output: %s", text[:200])
    return {}
