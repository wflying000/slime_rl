"""Semantic LLM judge for content-constrained instruction-following rewards."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .rule_content_reward import RuleContentResult

IF_BENCH_JUDGE_BASE_URL = os.getenv("IF_BENCH_JUDGE_BASE_URL", "")
IF_BENCH_JUDGE_API_KEY = os.getenv(
    "IF_BENCH_JUDGE_API_KEY",
    os.getenv("OPENAI_API_KEY", "EMPTY"),
)
IF_BENCH_JUDGE_MODEL = os.getenv("IF_BENCH_JUDGE_MODEL", "")
IF_BENCH_JUDGE_MAX_TOKENS = int(os.getenv("IF_BENCH_JUDGE_MAX_TOKENS", "512"))
IF_BENCH_JUDGE_TEMPERATURE = float(os.getenv("IF_BENCH_JUDGE_TEMPERATURE", "0.0"))
IF_BENCH_JUDGE_CONCURRENCY = int(os.getenv("IF_BENCH_JUDGE_CONCURRENCY", "128"))
IF_BENCH_JUDGE_TIMEOUT_S = float(
    os.getenv("IF_BENCH_JUDGE_TIMEOUT_S", os.getenv("IF_BENCH_JUDGE_TIMEOUT_SECONDS", "120"))
)
IF_BENCH_JUDGE_MAX_RETRIES = int(os.getenv("IF_BENCH_JUDGE_MAX_RETRIES", "1"))

ContentJudgeResult = int | float | bool | str | None
ContentJudgeFn = Callable[
    [str, str, dict[str, Any]],
    ContentJudgeResult | Awaitable[ContentJudgeResult],
]
IF_BENCH_CONTENT_JUDGE_FN: ContentJudgeFn | None = None

_ASYNC_OPENAI_CLIENT: Any | None = None
_ASYNC_OPENAI_CLIENT_CONFIG: tuple[str, str, float, int, int] | None = None
_JUDGE_SEMAPHORE: asyncio.Semaphore | None = None
_JUDGE_SEMAPHORE_CONFIG: tuple[int, int] | None = None


CONTENT_JUDGE_PROMPT = """You are an impartial and lenient content-presence judge. Hard formatting constraints have already been verified separately; your task is to evaluate ONLY whether the model response contains substantive content that is at least minimally related to the user's instruction.

Do NOT evaluate formatting requirements. Ignore whether the response follows required structure, JSON or XML format, bullet lists, exact word counts, capitalization, required keywords, ordering, or any other formatting constraint. Formatting is scored separately by rules.

Be lenient. When in doubt, assign [[1]]. Do not require the response to fully satisfy the instruction, satisfy all constraints, answer every sub-question, be factually perfect, be complete, or be high quality.

[Criteria for [[0]] (Fail)]
- The response is empty or nearly empty.
- The response is meaningless, garbled, random symbols, repeated tokens, or repeated phrases without real content.
- The response only appears to satisfy formatting requirements, such as outputting required tags, headings, keywords, placeholders, list markers, or structural elements, but contains no substantive content.
- The response is completely unrelated to the instruction and contains no meaningful attempt to address it.

[Criteria for [[1]] (Pass)]
- The response contains substantive natural-language content and has at least a little relevance to the instruction.
- The response makes any meaningful attempt to address the instruction, even if it is incomplete, shallow, partially wrong, or only weakly related.
- The response may miss many requirements and should still receive [[1]] if it contains meaningful content connected to the instruction.
- If the response has both formatting artifacts and some real content, assign [[1]].

[User instruction]
{instruction}

[Model response]
{response}

Please provide a concise assessment under 100 words. In your assessment:
1. State whether the response contains substantive content.
2. State whether it has at least minimal relevance to the instruction.
3. Only assign [[0]] if the response is meaningless, completely unrelated, or merely format-satisfying without real content.

End your output with the final score on a new line. Use EXACTLY this format:

Assessment: <Your concise reasoning here>
Score: <[[1]] or [[0]]>"""


def _build_content_judge_prompt(instruction: str, response: str) -> str:
    return CONTENT_JUDGE_PROMPT.format(instruction=instruction, response=response)


def _parse_judge_score(result: ContentJudgeResult) -> float:
    """Parse the judge's final score tag into a binary reward."""
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    if isinstance(result, (int, float)):
        return 1.0 if float(result) > 0 else 0.0
    if not result:
        return 0.0

    score_tags = re.findall(r"\[\[\s*([01](?:\.\d+)?)\s*\]\]", str(result).strip())
    if score_tags:
        return 1.0 if float(score_tags[-1]) > 0 else 0.0
    return 0.0


def _get_judge_concurrency(args: Any) -> int:
    value = getattr(args, "if_bench_judge_concurrency", None) if args is not None else None
    return IF_BENCH_JUDGE_CONCURRENCY if value is None else int(value)


def _get_judge_timeout_s(args: Any) -> float:
    value = getattr(args, "if_bench_judge_timeout_s", None) if args is not None else None
    if value is None and args is not None:
        value = getattr(args, "if_bench_judge_timeout_seconds", None)
    return IF_BENCH_JUDGE_TIMEOUT_S if value is None else float(value)


def _get_judge_max_retries(args: Any) -> int:
    value = getattr(args, "if_bench_judge_max_retries", None) if args is not None else None
    return IF_BENCH_JUDGE_MAX_RETRIES if value is None else int(value)


def _get_judge_semaphore(args: Any) -> asyncio.Semaphore | None:
    limit = _get_judge_concurrency(args)
    if limit <= 0:
        return None

    global _JUDGE_SEMAPHORE, _JUDGE_SEMAPHORE_CONFIG
    config = (limit, id(asyncio.get_running_loop()))
    if _JUDGE_SEMAPHORE is None or _JUDGE_SEMAPHORE_CONFIG != config:
        _JUDGE_SEMAPHORE = asyncio.Semaphore(limit)
        _JUDGE_SEMAPHORE_CONFIG = config
    return _JUDGE_SEMAPHORE


def _get_openai_message_field(message: Any, field: str) -> Any:
    if isinstance(message, dict):
        return message.get(field)

    value = getattr(message, field, None)
    if value is not None:
        return value

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and field in model_extra:
        return model_extra[field]

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped.get(field)
    return None


def _openai_message_text(message: Any) -> str | None:
    content = _get_openai_message_field(message, "content")
    if content:
        return str(content)

    reasoning_content = _get_openai_message_field(message, "reasoning_content")
    if reasoning_content:
        return str(reasoning_content)

    if isinstance(message, dict):
        return json.dumps(message, ensure_ascii=False)

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        try:
            return json.dumps(model_dump(), ensure_ascii=False)
        except Exception:
            return None
    return None


async def _openai_content_judge_async(args: Any, user_prompt: str) -> str | None:
    """Query an OpenAI-compatible judge endpoint without blocking the event loop."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required unless args.if_bench_content_judge_fn is provided"
        ) from exc

    base_url = getattr(args, "if_bench_judge_base_url", None) or IF_BENCH_JUDGE_BASE_URL
    model = getattr(args, "if_bench_judge_model", None) or IF_BENCH_JUDGE_MODEL
    if not base_url or not model:
        raise ValueError(
            "Content reward requires args.if_bench_content_judge_fn or both "
            "IF_BENCH_JUDGE_BASE_URL and IF_BENCH_JUDGE_MODEL"
        )

    global _ASYNC_OPENAI_CLIENT, _ASYNC_OPENAI_CLIENT_CONFIG
    api_key = getattr(args, "if_bench_judge_api_key", None) or IF_BENCH_JUDGE_API_KEY
    timeout_s = _get_judge_timeout_s(args)
    max_retries = _get_judge_max_retries(args)
    client_config = (
        base_url,
        api_key,
        timeout_s,
        max_retries,
        id(asyncio.get_running_loop()),
    )
    if _ASYNC_OPENAI_CLIENT is None or _ASYNC_OPENAI_CLIENT_CONFIG != client_config:
        _ASYNC_OPENAI_CLIENT = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        _ASYNC_OPENAI_CLIENT_CONFIG = client_config

    completion = await asyncio.wait_for(
        _ASYNC_OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=float(
                getattr(args, "if_bench_judge_temperature", None)
                or IF_BENCH_JUDGE_TEMPERATURE
            ),
            max_tokens=int(
                getattr(args, "if_bench_judge_max_tokens", None)
                or IF_BENCH_JUDGE_MAX_TOKENS
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        timeout=timeout_s,
    )
    return _openai_message_text(completion.choices[0].message)


def _resolve_content_judge(args: Any) -> ContentJudgeFn:
    judge_fn = getattr(args, "if_bench_content_judge_fn", None) if args is not None else None
    judge_fn = judge_fn or IF_BENCH_CONTENT_JUDGE_FN
    if judge_fn is not None:
        if not callable(judge_fn):
            raise TypeError("args.if_bench_content_judge_fn must be callable")
        return judge_fn

    def judge(
        instruction: str,
        response: str,
        metadata: dict[str, Any],
    ) -> Awaitable[str | None]:
        del metadata
        return _openai_content_judge_async(
            args,
            _build_content_judge_prompt(instruction, response),
        )

    return judge


async def _call_content_judge_async(
    args: Any,
    judge_fn: ContentJudgeFn,
    instruction: str,
    response: str,
    metadata: dict[str, Any],
) -> ContentJudgeResult:
    async def call() -> ContentJudgeResult:
        if inspect.iscoroutinefunction(judge_fn):
            result = judge_fn(instruction, response, metadata)
        else:
            result = await asyncio.to_thread(judge_fn, instruction, response, metadata)
        if inspect.isawaitable(result):
            result = await result
        return result

    semaphore = _get_judge_semaphore(args)
    if semaphore is None:
        return await call()

    async with semaphore:
        return await call()


async def llm_judge(
    response: str,
    instruction: str,
    metadata: dict[str, Any],
    args: Any = None,
    rule_result: RuleContentResult | dict[str, Any] | None = None,
) -> tuple[float, str | None]:
    """Score whether the response contains content relevant to the instruction."""
    judge_metadata = dict(metadata or {})
    if isinstance(rule_result, RuleContentResult):
        judge_metadata["rule_content_result"] = rule_result.to_dict()
    elif isinstance(rule_result, dict):
        judge_metadata["rule_content_result"] = rule_result

    judge_fn = _resolve_content_judge(args)
    judge_response = await _call_content_judge_async(
        args,
        judge_fn,
        instruction,
        response,
        judge_metadata,
    )
    return _parse_judge_score(judge_response), (
        None if judge_response is None else str(judge_response)
    )
