from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT_DIR / "code"
DATA_DIR = ROOT_DIR / "data"
PROMPT_DIR = ROOT_DIR / "prompts" / "part2-interactive-llm-forecasting"
RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = ROOT_DIR / "logs"
CHAT_LOG_DIR = LOG_DIR / "chat-logs" / "part2-interactive-llm-forecasting"
MASTER_LOG_DIR = LOG_DIR / "master-logs"
SYSTEM_LOG_DIR = LOG_DIR / "system-logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "part2-interactive-llm-forecasting"
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_API_KEYS_PATH = CONFIG_DIR / "api_keys.json"
DEFAULT_MODEL_SETUP_PATH = RESULTS_DIR / "part1-llm-strategic-consultation" / "part2-model-setup.csv"
