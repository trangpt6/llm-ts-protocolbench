from __future__ import annotations

import argparse
import json
import sys
import time

# Import from part2_api using importlib due to package directory naming
from part2_api import api_client, logging_utils, paths

ClientManager = api_client.ClientManager
setup_logger = logging_utils.setup_logger
DEFAULT_API_KEYS_PATH = paths.DEFAULT_API_KEYS_PATH
SYSTEM_LOG_DIR = paths.SYSTEM_LOG_DIR

API_TEST_DIR = SYSTEM_LOG_DIR / "api_test_logs"
logger = setup_logger("test_part2_apis", API_TEST_DIR)

TEST_MESSAGES = [
    {"role": "system", "content": "You are a concise time series forecasting assistant."},
    {
        "role": "user",
        "content": "In exactly two sentences, define one-step-ahead forecasting. End with [TEST_OK].",
    },
]

DETERMINISM_MESSAGES = [
    {"role": "user", "content": "What is 17 * 23? Reply with only the number."},
]

STATUS_LABELS = {
    "OK":           "OK",
    "INVALID_KEY":  "INVALID_KEY",
    "RATE_LIMITED": "RATE_LIMITED",
    "TIMEOUT":      "TIMEOUT",
    "CONFIG_ERROR": "CONFIG_ERROR",
    "FAIL":         "FAIL",
}

def classify_exception(exc: Exception) -> tuple[str, str]:
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "invalid api key", "unauthorized", "authentication")):
        return "INVALID_KEY", "API key is invalid or rejected"
    if any(k in msg for k in ("429", "rate limit", "quota", "too many requests")):
        return "RATE_LIMITED", "API key has exceeded its rate limit or quota"
    if any(k in msg for k in ("timeout", "timed out", "connect timeout", "read timeout")):
        return "TIMEOUT", "Connection to server timed out"
    if any(k in msg for k in ("400", "bad request", "model not found", "model_not_found")):
        return "INVALID_KEY", f"Invalid request - check model_id: {exc}"
    return "FAIL", str(exc)

def test_provider(client, provider: str) -> dict:
    result = {
        "provider": provider,
        "model_id": client.model_id,
        "status":   "UNKNOWN",
    }
    try:
        started  = time.time()
        response = client.chat(TEST_MESSAGES, max_tokens=300)
        latency  = round(time.time() - started, 3)
    except Exception as exc:
        status, detail = classify_exception(exc)
        result.update({"status": status, "error": detail})
        return result

    try:
        r1 = client.chat(DETERMINISM_MESSAGES, max_tokens=20).strip()
        r2 = client.chat(DETERMINISM_MESSAGES, max_tokens=20).strip()
    except Exception as exc:
        status, detail = classify_exception(exc)
        result.update({
            "status":           "OK",
            "latency_s":        latency,
            "has_test_ok":      "[TEST_OK]" in response,
            "response_preview": response[:300],
            "determinism_note": f"Determinism check failed: {detail}",
        })
        return result

    result.update({
        "status":           "OK",
        "latency_s":        latency,
        "has_test_ok":      "[TEST_OK]" in response,
        "response_preview": response[:300],
        "determinism_run1": r1,
        "determinism_run2": r2,
        "determinism_same": r1 == r2,
        "matches_expected": "391" in r1,
    })
    return result


def print_result(result: dict) -> None:
    status = result["status"]
    pid    = result["provider"]
    mid    = result["model_id"]

    if status == "OK":
        lat = result.get("latency_s", "?")
        tok = "OK" if result.get("has_test_ok") else "MISSING"
        det = "SAME" if result.get("determinism_same") else "DIFF"
        exp = "CORRECT" if result.get("matches_expected") else "WRONG"
        print(f"  [{pid}] {mid}")
        print(f"     Latency: {lat}s | [TEST_OK] tag: {tok} | Determinism: {det} | 17x23=391: {exp}")
        if "determinism_note" in result:
            print(f"     WARNING: {result['determinism_note']}")
    else:
        err = result.get("error", "")
        print(f"  [{pid}] {mid}")
        print(f"     {status}: {err}")

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("API PROVIDER HEALTH CHECK - PART 2")
    print("=" * 60)
    print()

    try:
        manager = ClientManager(args.api_keys)
    except Exception as exc:
        logger.error("Cannot load API providers: %s", exc)
        print(f"  CONFIG ERROR: {exc}")
        sys.exit(1)

    providers = args.provider or manager.list_providers()
    print(f"Found {len(providers)} provider(s): {', '.join(providers)}\n")

    results = []
    for provider in providers:
        print(f"Testing [{provider}]...", end=" ", flush=True)
        try:
            client = manager.get(provider)
        except Exception as exc:
            _, detail = "CONFIG_ERROR", str(exc)
            res = {"provider": provider, "model_id": "N/A", "status": "CONFIG_ERROR", "error": detail}
            print("CONFIG_ERROR")
            print(f"     {detail}")
            results.append(res)
            continue

        logger.info("Testing provider=%s model=%s", provider, client.model_id)
        result = test_provider(client, provider)
        results.append(result)

        print(result["status"])
        print_result(result)
        print()

        logger.info(
            "Result provider=%s status=%s latency=%s",
            provider,
            result["status"],
            result.get("latency_s", "N/A"),
        )

        if args.sleep:
            time.sleep(args.sleep)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"  Working (OK):           {counts.get('OK', 0)}")
    print(f"  Invalid key:            {counts.get('INVALID_KEY', 0)}")
    print(f"  Rate limited:           {counts.get('RATE_LIMITED', 0)}")
    print(f"  Timeout:                {counts.get('TIMEOUT', 0)}")
    print(f"  Config error:           {counts.get('CONFIG_ERROR', 0)}")
    print(f"  Other errors:           {counts.get('FAIL', 0)}")
    print(f"  Total:                  {len(results)}")
    print()

    output_path = API_TEST_DIR / f"api_test_results_{int(time.time())}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detailed results saved to: {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test LLM API providers for Part 2.")
    parser.add_argument("--provider", action="append", help="Provider to test. Repeat or omit for all configured.")
    parser.add_argument("--api-keys", type=lambda p: __import__("pathlib").Path(p), default=DEFAULT_API_KEYS_PATH)
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

# "model_id": "gemini-3.1-pro-preview",
# "display_name": "Gemini31Pro",
