import requests
import time
import json
import logging
import uuid
import traceback

from event_parsing import get_blocks, extract_content
from keys import CAPITOL_API_KEY
from default_payload import DEFAULT_PAYLOAD

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CAPITOL_URL = "https://hackathon.capitol.ai"


def _call_llm_endpoint(
    user_config_params: dict, external_id: str | None = None
) -> dict:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    llm_endpoint_payload = {
        "params": {"external_id": external_id or str(uuid.uuid4())},
        "user_config_params": user_config_params,
    }
    response = requests.post(
        f"{CAPITOL_URL}/llm", headers=headers, json=llm_endpoint_payload
    )
    try:
        response_json = response.json()
        if "external_id" not in response_json:
            logger.debug(
                "External ID missing",
                extra={
                    "info_context": {
                        "response_json": response_json,
                        "api_key": CAPITOL_API_KEY,
                    }
                },
            )
            return {}
        return response_json
    except Exception as e:
        logger.debug(
            "Failed to generate document",
            extra={
                "info_context": {
                    "status_code": response.status_code,
                    "external_id": llm_endpoint_payload["params"]["external_id"],
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                    "api_key": CAPITOL_API_KEY,
                }
            },
        )
        return {}


def _is_complete(external_id: str) -> bool:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    response = requests.get(f"{CAPITOL_URL}/attributes/{external_id}", headers=headers)
    if response.status_code != 200:
        logger.warning("Failed to poll document generation")
        return False
    response_json = response.json()
    return not response_json["is_generating"]


def _wait_until_complete(external_id: str, timeout_minutes: int = 5) -> bool:
    duration_seconds = timeout_minutes * 60
    end_time = time.monotonic() + duration_seconds
    while time.monotonic() < end_time:
        if _is_complete(external_id):
            return True
        time.sleep(1)
    return False


def _get_events(external_id: str) -> list[dict]:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    response = requests.get(
        f"{CAPITOL_URL}/events", headers=headers, params={"external_id": external_id}
    )
    response_json = response.json()
    events = response_json["events"]
    return events


def run_payload(
    payload: dict,
    timeout_minutes: int = 10,
    overrides: dict | None = None,
    id: str | None = None,
) -> dict[str, list[dict]]:
    overrides = overrides or {}
    response_json = _call_llm_endpoint({**payload, **overrides})
    if response_json:
        external_id = response_json["external_id"]
        # draft_id = response_json["draft_id"]
        if not _wait_until_complete(external_id, timeout_minutes):
            logger.warning(
                f"Document generation did not complete in {timeout_minutes} minutes.",
                extra={"info_context": {"id": id}},
            )
            return {}
        logger.info(
            "Document generation completed.",
            extra={"info_context": {"id": id}},
        )
        events = _get_events(external_id)
        blocks = [extract_content(block) for block in get_blocks(events)]
        return blocks
    else:
        return None


def test_api():
    payload = {**DEFAULT_PAYLOAD, **{"user_query": "explain string theory"}}
    blocks = run_payload(payload)
    print(json.dumps(blocks, indent=4))


if __name__ == "__main__":
    test_api()
