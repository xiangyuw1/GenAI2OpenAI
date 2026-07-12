import argparse
import json
import logging
import os
import sys

from auth.cas_login import LoginError
from auth.token_manager import TokenManager
from config import Config
from app import create_app

parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, required=True,
                    help='JWT token (eyJ...) or student_id@password for auto-login')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging')
parser.add_argument('--api-key', type=str, default=None,
                    help='API key for client authentication (or set API_KEY env var)')
parser.add_argument('--disable-fallback-renew', action='store_true',
                    help='Disable automatic JWT renew+retry when upstream returns "Token失效" '
                         '(only applies to credential token mode)')
parser.add_argument('--claude-haiku-model', type=str, default=os.environ.get("CLAUDE_HAIKU_MODEL", "qwen-instruct"),
                    help='GenAI model mapped from Claude haiku requests')
parser.add_argument('--claude-sonnet-model', type=str, default=os.environ.get("CLAUDE_SONNET_MODEL", "gpt-4.1"),
                    help='GenAI model mapped from Claude sonnet requests')
parser.add_argument('--claude-opus-model', type=str, default=os.environ.get("CLAUDE_OPUS_MODEL", "gpt-5.5"),
                    help='GenAI model mapped from Claude opus requests')
parser.add_argument('--config', type=str, default=os.environ.get("CONFIG", ""),
                    help='JSON like {"mapping":{"gpt-5-codex":"deepseek-pro"}}')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

_config_json = {}
if args.config and args.config.strip():
    try:
        _config_json = json.loads(args.config)
    except json.JSONDecodeError as e:
        logger.error("Invalid --config JSON: %s", e)
        sys.exit(1)

try:
    token_manager = TokenManager(args.token)
    token_manager.initial_login()
except (LoginError, ValueError) as e:
    logger.error("Failed to initialize token: %s", e)
    sys.exit(1)

config = Config(
    token_manager=token_manager,
    port=args.port,
    api_key=args.api_key or os.environ.get("API_KEY"),
    debug=args.debug,
    fallback_renew=not args.disable_fallback_renew,
    claude_haiku_model=args.claude_haiku_model,
    claude_sonnet_model=args.claude_sonnet_model,
    claude_opus_model=args.claude_opus_model,
    model_mapping=_config_json.get("mapping", {}),
)

app = create_app(config)

if __name__ == '__main__':
    logger.info("Starting GenAI2OpenAI proxy on 127.0.0.1:%d", config.port)
    logger.info("Debug: %s, Auth: %s, Token mode: %s",
                config.debug, "enabled" if config.api_key else "disabled",
                token_manager.mode)
    app.run(host='127.0.0.1', port=config.port, debug=False)
