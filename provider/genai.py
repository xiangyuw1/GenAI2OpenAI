import itertools
import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime

import requests

from auth.cas_login import LoginError
from auth.token_manager import TokenExpiredError
from config import (
    GENAI_READ_TIMEOUT,
    GENAI_REQUEST_TIMEOUT,
    GENAI_URL,
    build_genai_headers,
    model_registry,
)
from errors import make_error_chunk


_TOKEN_EXPIRED_MARKERS = ("Token失效", "token失效", "请重新登录", "登录已过期")


def _try_parse_sse_line(raw):
    if raw is None:
        return None
    s = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if s.startswith("data:"):
        s = s[5:].strip()
    if not s or s == "[DONE]":
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _is_upstream_token_expired(genai_json):
    if not isinstance(genai_json, dict) or genai_json.get("success") is not False:
        return False
    msg = genai_json.get("message") or ""
    if any(marker in msg for marker in _TOKEN_EXPIRED_MARKERS):
        return True
    lower = msg.lower()
    return "token" in lower and ("expired" in lower or "invalid" in lower)


def _is_upstream_business_error(genai_json):
    if not isinstance(genai_json, dict):
        return False
    if genai_json.get("success") is False:
        return True
    code = genai_json.get("code")
    if isinstance(code, int) and code >= 400:
        return True
    if "errMsg" in genai_json or "message" in genai_json:
        if isinstance(code, int) and code >= 400:
            return True
        if "errMsg" in genai_json and genai_json.get("errMsg"):
            return True
    return False


def _upstream_business_error_message(genai_json):
    if not isinstance(genai_json, dict):
        return "Upstream error"
    return (
        genai_json.get("errMsg")
        or genai_json.get("message")
        or genai_json.get("err_msg")
        or "Upstream error"
    )
from model_config.registry import (
    apply_model_mapping,
    get_genai_id,
    get_root_ai_type,
    resolve_model,
    supports_reasoning,
)
from tools.parsing import tag_prefix_len

logger = logging.getLogger(__name__)

POST_FINISH_DRAIN_MAX_LINES = 20
POST_FINISH_DRAIN_TIMEOUT_SECONDS = 1.0


def _coerce_int(value):
    """Return a non-negative int for token counters, or None when unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _first_int(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = _coerce_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _usage_candidates(response_data):
    if not isinstance(response_data, dict):
        return []

    candidates = []
    for key in ("usage", "tokenUsage", "token_usage", "tokens", "stats", "data", "result"):
        value = response_data.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    other = response_data.get("other")
    if isinstance(other, dict):
        candidates.append(other)
    elif isinstance(other, str):
        try:
            parsed_other = json.loads(other)
        except json.JSONDecodeError:
            parsed_other = None
        if isinstance(parsed_other, dict):
            candidates.append(parsed_other)

    candidates.append(response_data)
    return candidates


def _nested_int(mapping, parent_key, *keys):
    if not isinstance(mapping, dict):
        return None
    nested = mapping.get(parent_key)
    if not isinstance(nested, dict):
        return None
    return _first_int(nested, *keys)


def extract_usage_from_genai(response_data):
    """Normalize GenAI/OpenAI-style token usage fields to OpenAI usage shape."""
    prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = total_tokens = None

    for candidate in _usage_candidates(response_data):
        prompt_tokens = prompt_tokens if prompt_tokens is not None else _first_int(
            candidate,
            "prompt_tokens", "promptTokens", "input_tokens", "inputTokens",
            "inputTokenCount", "promptTokenCount", "prompt_token_count",
        )
        completion_tokens = completion_tokens if completion_tokens is not None else _first_int(
            candidate,
            "completion_tokens", "completionTokens", "output_tokens", "outputTokens",
            "outputTokenCount", "completionTokenCount", "completion_token_count",
            "generated_tokens", "generatedTokens",
        )
        reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else _first_int(
            candidate,
            "reasoning_tokens", "reasoningTokens", "thinking_tokens", "thinkingTokens",
            "reasoningTokenCount", "thinkingTokenCount",
        )
        cached_tokens = cached_tokens if cached_tokens is not None else _first_int(
            candidate,
            "cached_tokens", "cachedTokens", "cache_tokens", "cacheTokens",
            "cachedTokenCount", "cacheTokenCount",
        )
        total_tokens = total_tokens if total_tokens is not None else _first_int(
            candidate,
            "total_tokens", "totalTokens", "totalTokenCount", "tokens_total", "tokensTotal",
        )

        reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else _nested_int(
            candidate, "completion_tokens_details", "reasoning_tokens", "reasoningTokens"
        )
        cached_tokens = cached_tokens if cached_tokens is not None else _nested_int(
            candidate, "prompt_tokens_details", "cached_tokens", "cachedTokens"
        )

    if all(value is None for value in (
        prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, total_tokens,
    )):
        return None

    usage = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}

    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    elif completion_tokens is not None or reasoning_tokens is not None:
        total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if completion_tokens is None and reasoning_tokens is not None:
            total_tokens += reasoning_tokens
        usage["total_tokens"] = total_tokens
    return usage


def estimate_token_count(text):
    """Cheap no-dependency token estimate used only when upstream omits usage."""
    if not text:
        return 0
    cjk = 0
    other_chars = 0
    for char in str(text):
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
        elif not char.isspace():
            other_chars += 1
    estimate = (cjk * 2) + ((other_chars + 3) // 4)
    return max(1, estimate)


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def estimate_messages_token_count(messages):
    total = 0
    for message in messages or []:
        if not isinstance(message, dict):
            total += estimate_token_count(str(message))
            continue
        total += 4  # small per-message chat framing overhead
        total += estimate_token_count(message.get("role", ""))
        total += estimate_token_count(_content_to_text(message.get("content")))
        if message.get("name"):
            total += estimate_token_count(message["name"])
        if message.get("tool_calls"):
            total += estimate_token_count(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return total


def _usage_detail_value(usage, detail_key, counter_key):
    details = usage.get(detail_key) if isinstance(usage, dict) else None
    if isinstance(details, dict):
        return _coerce_int(details.get(counter_key))
    return None


def _usage_detail_dict(usage, detail_key):
    details = usage.get(detail_key) if isinstance(usage, dict) else None
    return dict(details) if isinstance(details, dict) else {}


def complete_usage(usage, *, prompt_tokens=None, completion_tokens=None, reasoning_tokens=None, cached_tokens=None):
    """Fill missing usage counters and recompute total when necessary."""
    result = dict(usage or {})
    filled_completion_tokens = False

    if "prompt_tokens" not in result and prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    if "completion_tokens" not in result and completion_tokens is not None:
        result["completion_tokens"] = completion_tokens
        filled_completion_tokens = True

    existing_reasoning = _usage_detail_value(result, "completion_tokens_details", "reasoning_tokens")
    if existing_reasoning is None and reasoning_tokens is not None:
        details = _usage_detail_dict(result, "completion_tokens_details")
        details["reasoning_tokens"] = reasoning_tokens
        result["completion_tokens_details"] = details

    existing_cached = _usage_detail_value(result, "prompt_tokens_details", "cached_tokens")
    if existing_cached is None and cached_tokens is not None:
        details = _usage_detail_dict(result, "prompt_tokens_details")
        details["cached_tokens"] = cached_tokens
        result["prompt_tokens_details"] = details

    prompt = _coerce_int(result.get("prompt_tokens")) or 0
    completion = _coerce_int(result.get("completion_tokens")) or 0
    reasoning = _usage_detail_value(result, "completion_tokens_details", "reasoning_tokens") or 0

    if "total_tokens" not in result:
        # OpenAI usually includes reasoning in completion_tokens. If we had to
        # estimate completion from visible text, add estimated reasoning too.
        result["total_tokens"] = prompt + completion + (reasoning if filled_completion_tokens else 0)

    return result


def _drain_post_finish_lines(
    line_iter,
    *,
    max_lines=POST_FINISH_DRAIN_MAX_LINES,
    timeout=POST_FINISH_DRAIN_TIMEOUT_SECONDS,
    close=None,
):
    """Read post-finish accounting lines in a daemon reader, bounded by time/lines."""
    drained = queue.Queue()
    sentinel = object()

    def reader():
        try:
            for _, item in zip(range(max_lines), line_iter):
                drained.put(item)
        except Exception as exc:  # pragma: no cover - defensive for socket teardown
            drained.put(exc)
        finally:
            drained.put(sentinel)

    thread = threading.Thread(target=reader, name="genai-usage-drain", daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout
    lines_seen = 0
    while lines_seen < max_lines:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = drained.get(timeout=min(0.05, remaining))
        except queue.Empty:
            if not thread.is_alive():
                break
            continue
        if item is sentinel:
            break
        if isinstance(item, Exception):
            logger.debug("Post-finish usage drain stopped: %s", item)
            break
        lines_seen += 1
        yield item

    if thread.is_alive() and close is not None:
        try:
            close()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to close GenAI response after usage drain timeout: %s", exc)


def _extract_text_from_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts.append(part.get("text", ""))
        return " ".join(parts).strip()
    return str(content) if content else ""


def _content_has_images(content):
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and part.get("type") in ("image_url", "input_image")
            for part in content
        )
    return False


def _flatten_messages_text_only(messages):
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    text_parts.append(part.get("text", ""))
            new_msg = dict(msg)
            new_msg["content"] = " ".join(text_parts).strip()
            result.append(new_msg)
        else:
            result.append(msg)
    return result


def convert_messages_to_genai_format(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text_from_content(msg.get("content", ""))
    return ""


def extract_content_from_genai(response_data):
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            content = delta.get("content") or None
            reasoning = delta.get("reasoning_content") or None
            return content, reasoning
    except (KeyError, IndexError, TypeError):
        pass
    return None, None


def stream_genai_response(chat_info, messages, model, max_tokens, config):
    token = config.token_manager.get_token()
    model = apply_model_mapping(model, getattr(config, "model_mapping", {})) or model
    genai_id = get_genai_id(model)
    record = None
    if not resolve_model(model):
        model_info = model_registry.get_models(token).get(model)
        if model_info:
            record = {"rootAiType": model_info.root_ai_type}
    root_ai_type = get_root_ai_type(model, genai_record=record)

    has_image_content = any(
        _content_has_images(msg.get("content", ""))
        for msg in messages if msg.get("role") == "user"
    )
    if has_image_content:
        logger.info("Image content detected in messages, rootAiType=%s", root_ai_type)

    if root_ai_type != "azure" and has_image_content:
        logger.info("Flattening messages to text-only for rootAiType=%s", root_ai_type)
        messages = _flatten_messages_text_only(messages)

    prompt_token_estimate = estimate_messages_token_count(messages)

    genai_data = {
        "chatInfo": chat_info,
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": genai_id,
        "aiSecType": "1",
        "promptTokens": prompt_token_estimate,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }

    logger.debug("=== GenAI Request ===")
    logger.debug("Model: %s (aiType=%s), rootAiType: %s", model, genai_id, root_ai_type)
    logger.debug("Messages count: %d", len(messages))
    for i, msg in enumerate(messages):
        role = msg.get('role', '?')
        content = msg.get('content', '')
        preview = (content[:200] + '...') if content and len(content) > 200 else content
        logger.debug("  [%d] role=%s, content=%s", i, role, preview)

    fallback_renew = (
        getattr(config, "fallback_renew", True)
        and getattr(config.token_manager, "mode", None) == "credential"
    )

    try:
        response = None
        first_line = None
        token_renew_attempted = False
        while True:
            headers = build_genai_headers(token)

            last_exc = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        GENAI_URL,
                        headers=headers,
                        json=genai_data,
                        stream=True,
                        timeout=GENAI_REQUEST_TIMEOUT
                    )
                    last_exc = None
                    break
                except requests.exceptions.ConnectionError as e:
                    last_exc = e
                    if attempt < 2:
                        wait = 2 ** attempt
                        logger.warning(
                            "Connection error (attempt %d/3), retrying in %ds: %s",
                            attempt + 1, wait, str(e)[:120]
                        )
                        time.sleep(wait)
            if last_exc is not None:
                raise last_exc

            logger.debug("GenAI Response Status: %d", response.status_code)

            if response.status_code == 401 and not token_renew_attempted:
                new_token = config.token_manager.force_refresh()
                if new_token:
                    logger.info("Token refreshed after 401, retrying request")
                    token = new_token
                    token_renew_attempted = True
                    response.close()
                    continue

            if response.status_code != 200:
                logger.warning("GenAI API error %d: %s", response.status_code, response.text[:500])
                if response.status_code == 401:
                    yield make_error_chunk("Upstream authentication failed", model)
                elif response.status_code == 429:
                    yield make_error_chunk("Upstream rate limit exceeded", model)
                else:
                    yield make_error_chunk(f"Upstream API error: {response.status_code}", model)
                return

            # Peek the first non-empty SSE line to detect upstream-side token
            # rejection BEFORE we start yielding chunks. The "Token失效" error
            # is reported in-band as the first SSE line.
            line_iter = response.iter_lines()
            first_line = None
            for raw in line_iter:
                if raw:
                    first_line = raw
                    break

            first_json = _try_parse_sse_line(first_line)
            if _is_upstream_token_expired(first_json):
                err_msg = first_json.get("message", "Token expired")
                if fallback_renew and not token_renew_attempted:
                    logger.warning(
                        "Upstream rejected JWT (%s); force-refreshing and retrying once",
                        err_msg,
                    )
                    response.close()
                    try:
                        new_token = config.token_manager.force_refresh()
                    except (LoginError, TokenExpiredError) as e:
                        logger.warning("Force refresh failed during fallback renew: %s", e)
                        yield make_error_chunk(f"Token renew failed: {e}", model)
                        return
                    if not new_token:
                        yield make_error_chunk("Token renew returned empty token", model)
                        return
                    token = new_token
                    token_renew_attempted = True
                    continue
                else:
                    reason = (
                        "fallback renew disabled" if not fallback_renew
                        else "already retried once"
                    )
                    logger.warning(
                        "Upstream token expired (%s); not retrying (%s)",
                        err_msg, reason,
                    )
                    yield make_error_chunk(f"Upstream error: {err_msg}", model)
                    return

            # Not a token-expired error → proceed to normal streaming.
            break

        finished = False
        finish_reason_seen = "stop"
        line_count = 0
        latest_usage = None
        completion_token_estimate = 0
        reasoning_token_estimate = 0
        abort_stream = False

        def terminal_chunk(finish_reason="stop"):
            usage = complete_usage(
                latest_usage,
                prompt_tokens=prompt_token_estimate,
                completion_tokens=completion_token_estimate,
                reasoning_tokens=reasoning_token_estimate or None,
            )
            final_response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }],
                "usage": usage,
            }
            return f"data: {json.dumps(final_response)}\n\n"

        def process_line(line, *, emit_content=True):
            nonlocal line_count, latest_usage, completion_token_estimate
            nonlocal reasoning_token_estimate, finished, finish_reason_seen, abort_stream

            if not line:
                return []

            line_str = line.decode('utf-8') if isinstance(line, bytes) else line

            if line_count < 5:
                logger.debug("Raw line [%d]: %s", line_count, line_str[:300])
            line_count += 1

            if line_str.startswith('data:'):
                line_str = line_str[5:].strip()

            if not line_str or line_str == "[DONE]":
                return []

            try:
                genai_json = json.loads(line_str)
            except json.JSONDecodeError as e:
                logger.debug("JSON decode error: %s, line: %s", e, line_str[:200])
                return []

            if _is_upstream_business_error(genai_json):
                err_msg = _upstream_business_error_message(genai_json)
                err_code = genai_json.get("code", 500)
                logger.warning("GenAI business error (code=%s): %s", err_code, err_msg)
                abort_stream = True
                return [make_error_chunk(f"Upstream error: {err_msg}", model)]

            usage = extract_usage_from_genai(genai_json)
            if usage:
                latest_usage = usage

            content, reasoning = extract_content_from_genai(genai_json)

            delta = {}
            if content:
                delta["content"] = content
                completion_token_estimate += estimate_token_count(content)
            if reasoning:
                delta["reasoning_content"] = reasoning
                reasoning_token_estimate += estimate_token_count(reasoning)

            chunks = []
            if emit_content and delta:
                openai_response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }
                chunks.append(f"data: {json.dumps(openai_response)}\n\n")

            if "choices" in genai_json and len(genai_json["choices"]) > 0:
                choice = genai_json["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    finished = True
                    finish_reason_seen = finish_reason or "stop"

            return chunks

        # Re-prepend the peeked first line so it's processed by the normal loop.
        if first_line is not None:
            line_iter = itertools.chain([first_line], line_iter)
        for line in line_iter:
            for chunk in process_line(line):
                yield chunk
            if abort_stream:
                return
            if finished:
                close = getattr(response, "close", None)
                for drain_line in _drain_post_finish_lines(line_iter, close=close):
                    for _ in process_line(drain_line, emit_content=False):
                        pass
                    if abort_stream:
                        return
                break

        logger.debug("Total lines received: %d, finished: %s", line_count, finished)

        yield terminal_chunk(finish_reason_seen)
        yield "data: [DONE]\n\n"

    except requests.exceptions.ReadTimeout:
        logger.warning("GenAI stream read timeout after %.1fs", GENAI_READ_TIMEOUT)
        yield make_error_chunk(
            f"Upstream GenAI read timed out after {GENAI_READ_TIMEOUT:g}s. "
            "The request may be too large or the upstream model may be busy.",
            model,
        )
    except requests.exceptions.ConnectTimeout:
        logger.warning("GenAI stream connect timeout")
        yield make_error_chunk("Upstream GenAI connection timed out", model)
    except requests.exceptions.RequestException as e:
        logger.exception("GenAI stream request failed")
        yield make_error_chunk(f"Upstream GenAI request failed: {e}", model)
    except Exception as e:
        logger.exception("Error in stream_genai_response")
        yield make_error_chunk(str(e), model)


def stream_genai_response_with_tools(
    chat_info,
    messages,
    model,
    max_tokens,
    config,
    adapter,
    tools=None,
):
    model = apply_model_mapping(model, getattr(config, "model_mapping", {})) or model
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    open_tag = adapter.open_tags[0] if adapter else "<tool_call>"

    buffer = ""
    tool_buffer = ""
    sent_role = False
    tool_detected = False
    think_enabled = supports_reasoning(model)
    think_state = "outside"
    think_buffer = ""
    latest_usage = None

    def make_chunk(delta, finish_reason=None, usage=None):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"

    def emit_text(text):
        nonlocal sent_role
        delta = {"content": text}
        if not sent_role:
            delta["role"] = "assistant"
            sent_role = True
        return make_chunk(delta)

    for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("usage"), dict):
            latest_usage = data["usage"]
        if "choices" not in data or not data["choices"]:
            continue
        chunk_delta = data["choices"][0].get("delta", {})
        content = chunk_delta.get("content", "")
        reasoning = chunk_delta.get("reasoning") or chunk_delta.get("reasoning_content", "")
        if reasoning:
            yield make_chunk({"reasoning_content": reasoning})
        if not content:
            continue

        if think_enabled:
            if think_state == "outside":
                if "<think>" in content:
                    before, _, after = content.partition("<think>")
                    if before:
                        content = before
                    else:
                        content = ""
                    think_state = "inside"
                    think_buffer = ""
                    if after:
                        content = content + after

            if think_state == "inside":
                if "</think>" in content:
                    before, _, after = content.partition("</think>")
                    think_buffer += before
                    if think_buffer:
                        yield make_chunk({"reasoning_content": think_buffer})
                    think_buffer = ""
                    think_state = "outside"
                    content = after
                else:
                    think_buffer += content
                    continue

        if not content:
            continue

        if tool_detected:
            tool_buffer += content
            continue

        buffer += content

        while True:
            tag_pos = buffer.find(open_tag)
            if tag_pos >= 0:
                pre = buffer[:tag_pos]
                if pre:
                    yield emit_text(pre)

                tool_detected = True
                tool_buffer = buffer[tag_pos:]
                buffer = ""
                break

            plen = tag_prefix_len(buffer, open_tag)
            if plen > 0:
                safe = buffer[:-plen]
                if safe:
                    yield emit_text(safe)
                buffer = buffer[-plen:]
                break

            if buffer:
                yield emit_text(buffer)
            buffer = ""
            break

    if tool_detected:
        result = adapter.extract_tool_calls(tool_buffer, tools=tools)
        tool_calls = result.tool_calls
        remaining = result.remaining_text

        if tool_calls:
            logger.debug("Streaming tool calling: detected %d tool_call(s)", len(tool_calls))

            if result.parse_errors:
                logger.warning("Tool call parse errors: %s", result.parse_errors)

            if remaining and remaining.strip():
                yield emit_text(remaining.strip())

            if buffer:
                yield emit_text(buffer)
                buffer = ""

            if not sent_role:
                yield make_chunk({"role": "assistant"})
                sent_role = True

            for i, tc in enumerate(tool_calls):
                yield make_chunk({
                    "tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }]
                })

            yield make_chunk({}, finish_reason="tool_calls", usage=latest_usage)
            yield "data: [DONE]\n\n"
        else:
            logger.warning("Tool tag detected but parsing failed — emitting as text")
            yield emit_text(tool_buffer)
            yield make_chunk({}, finish_reason="stop", usage=latest_usage)
            yield "data: [DONE]\n\n"
    else:
        if buffer:
            yield emit_text(buffer)

        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})

        yield make_chunk({}, finish_reason="stop", usage=latest_usage)
        yield "data: [DONE]\n\n"
