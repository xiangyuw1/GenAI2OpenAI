from __future__ import annotations

import logging
from model_config.spec import ModelSpec

logger = logging.getLogger(__name__)

MODEL_SPECS: dict[str, ModelSpec] = {
    "glm-5.1": ModelSpec(
        public_id="glm-5.1",
        genai_id="chatglm",
        root_ai_type="xinference",
        tool_adapter="glm",
    ),
    "gpt-4.1": ModelSpec(
        public_id="gpt-4.1",
        genai_id="GPT-4.1",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-4.1-mini": ModelSpec(
        public_id="gpt-4.1-mini",
        genai_id="GPT-4.1-mini",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-o4-mini": ModelSpec(
        public_id="gpt-o4-mini",
        genai_id="o4-mini",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-o3": ModelSpec(
        public_id="gpt-o3",
        genai_id="o3",
        root_ai_type="azure",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    # 实测：上海科大 GenAI 平台部署的 deepseek-chat/deepseek-pro 不识别 DSML 协议
    # （benchmark 里 deepseek_v4 适配器 0%，generic 92%）。默认走 generic，
    # 想试 DSML 仍可通过 model@deepseek_v4 后缀临时启用。
    "deepseek-v4-flash": ModelSpec(
        public_id="deepseek-v4-flash",
        genai_id="deepseek-chat",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "deepseek-v4-pro": ModelSpec(
        public_id="deepseek-v4-pro",
        genai_id="deepseek-pro",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "qwen-instruct": ModelSpec(
        public_id="qwen-instruct",
        genai_id="qwen-instruct",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    # 实测 minimax 适配器(88%)略低于 generic(90%)；保留专属适配器但默认 generic。
    "minimax-m1": ModelSpec(
        public_id="minimax-m1",
        genai_id="MiniMax-M1",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "gpt-5.5": ModelSpec(
        public_id="gpt-5.5",
        genai_id="GPT-5.5",
        root_ai_type="azure",
        tool_adapter="generic",
        supports_reasoning=True,
    )
}

_ALIAS_MAP: dict[str, ModelSpec] = {}
for _spec in MODEL_SPECS.values():
    _ALIAS_MAP[_spec.public_id.lower()] = _spec
    _ALIAS_MAP[_spec.genai_id.lower()] = _spec

_EXTRA_ALIASES: dict[str, str] = {
    "glm": "glm-5.1",
    "chatglm": "glm-5.1",
    "gpt4.1": "gpt-4.1",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen-instruct",
}
for _alias, _target in _EXTRA_ALIASES.items():
    spec = MODEL_SPECS.get(_target)
    if spec:
        _ALIAS_MAP[_alias.lower()] = spec


def parse_model_override(model_id: str | None) -> tuple[str | None, str | None]:
    """支持 ``model@adapter`` 语法以临时切换 tool adapter（用于 benchmark / 调试）。

    例如 ``deepseek-v4-flash@generic`` 表示用 deepseek-v4-flash 模型但强制使用
    generic 适配器。返回 ``(纯净 model_id, adapter_override 或 None)``。
    """
    if not model_id or "@" not in model_id:
        return model_id, None
    base, _, override = model_id.rpartition("@")
    base = base.strip() or model_id
    override = override.strip().lower() or None
    return base, override


def apply_model_mapping(model_id: str | None, mapping: dict[str, str] | None) -> str | None:
    if not model_id or not mapping:
        return model_id

    base, override = parse_model_override(model_id)
    mapped_base = mapping.get(base or "", base or model_id)
    if override:
        return f"{mapped_base}@{override}"
    return mapped_base


def resolve_model(model_id: str | None) -> ModelSpec | None:
    base, _ = parse_model_override(model_id)
    return _ALIAS_MAP.get((base or "").lower())


def get_genai_id(model_id: str) -> str:
    spec = resolve_model(model_id)
    if spec:
        return spec.genai_id
    
    # 尝试从 model_registry 中进行大小写不敏感的查找
    # 这处理动态模型（如 GPT-5.6-SOL）的大小写变体
    try:
        from config import model_registry
        # 尝试获取模型，如果未加载则触发加载
        models = model_registry._models
        if not models:
            # 尝试从 Flask context 获取 token 来加载模型
            try:
                from flask import g
                token = getattr(g, 'token', None)
                if token:
                    models = model_registry.get_models(token)
            except Exception:
                pass
        
        if models:
            model_id_lower = model_id.lower()
            for ai_type, info in models.items():
                if ai_type.lower() == model_id_lower:
                    logger.debug("Case-insensitive match: %s -> %s", model_id, ai_type)
                    return ai_type
    except Exception:
        pass
    
    return model_id


def get_root_ai_type(model_id: str, genai_record: dict | None = None) -> str:
    spec = resolve_model(model_id)
    if spec:
        return spec.root_ai_type
    if genai_record:
        return genai_record.get("rootAiType", "xinference")
    return "xinference"


def select_tool_adapter(model_id: str, genai_record: dict | None = None) -> str:
    _, override = parse_model_override(model_id)
    if override:
        return override
    spec = resolve_model(model_id)
    if spec:
        return spec.tool_adapter

    text = _build_model_text(model_id, genai_record)
    # 仅 GLM 保留专属适配器（实测有正向收益）；deepseek / minimax 实测无收益或负收益，
    # 走 generic。如需启用专属适配器，使用 model@deepseek_v4 / model@minimax 后缀。
    if "chatglm" in text or ("glm" in text and "xinference" in text):
        return "glm"
    return "generic"


def supports_reasoning(model_id: str) -> bool:
    spec = resolve_model(model_id)
    return spec.supports_reasoning if spec else False


def _build_model_text(model_id: str | None, record: dict | None) -> str:
    parts = [(model_id or "").lower()]
    if record:
        for key in ("aiType", "aiName", "rootModelName", "simpleName", "rootAiType"):
            value = record.get(key)
            if value:
                parts.append(str(value).lower())
    return " ".join(parts)
