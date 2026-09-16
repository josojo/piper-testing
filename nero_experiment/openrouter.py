"""Small OpenRouter client. Model output is untrusted until locally checked."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import urllib.error
import urllib.request

import numpy as np
from dotenv import load_dotenv

from nero_planner import PlanningError, Pose


DEFAULT_MODEL = "gpt-5.6-luna"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def strict_json(text):
    def bad_constant(value):
        raise PlanningError(f"Nonfinite JSON constant: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PlanningError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=bad_constant, object_pairs_hook=unique_pairs)


def response_schema(candidate_ids):
    def array(size):
        return {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}

    return {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "target", "reason"],
        "properties": {
            "candidate_id": {"type": "string", "enum": [*candidate_ids, "stop"]},
            "target": {"anyOf": [{"type": "null"}, {
                "type": "object", "additionalProperties": False,
                "required": ["frame", "position_m", "orientation_xyzw"],
                "properties": {"frame": {"type": "string", "enum": ["nero_base"]},
                               "position_m": array(3), "orientation_xyzw": array(4)},
            }]},
            "reason": {"type": "string"},
        },
    }


def validate_selection(value, candidates):
    if not isinstance(value, dict) or set(value) != {"candidate_id", "target", "reason"}:
        raise PlanningError("LLM must return candidate_id, target and reason only")
    if not isinstance(value["reason"], str) or len(value["reason"]) > 2000:
        raise PlanningError("LLM reason must be short text")
    if not isinstance(value["candidate_id"], str):
        raise PlanningError("candidate_id must be text")
    if value["candidate_id"] == "stop":
        if value["target"] is not None:
            raise PlanningError("stop must have a null target")
        raise RuntimeError("LLM declined the experiment: " + value["reason"])
    selected = next((c for c in candidates if c.id == value["candidate_id"]), None)
    if selected is None:
        raise PlanningError("LLM chose an unknown candidate")
    pose = Pose.from_dict(value["target"])
    expected = selected.trajectory.source_target_pose
    if (not np.allclose(pose.position_m, expected.position_m, atol=1e-7, rtol=0)
            or abs(np.dot(pose.orientation_xyzw, expected.orientation_xyzw)) < 1 - 1e-10):
        raise PlanningError("LLM coordinates do not match the locally validated reachable candidate")
    return selected


class OpenRouter:
    def __init__(self, model=DEFAULT_MODEL, api_key=None, timeout=60, transport=None):
        self.model = model
        # Load the project-root .env while preserving any explicit shell values.
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in the environment on the machine running this command")
        self.timeout = timeout
        self.transport = transport or urllib.request.urlopen

    def choose(self, context, candidates, correction=None):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You select the next locally validated Cartesian waypoint for a supervised seven-joint NERO experiment. "
                    "The task is to straighten the arm toward the supplied upright joint reference, then "
                    "return slowly to the original joint configuration. Only choose from the supplied "
                    "locally reachable, full-path validated candidates. Copy its ID and pose coordinates "
                    "exactly. Prefer the greatest joint progress that remains feasible within the "
                    "remaining pose-request budget, with good clearance. These waypoints are executed "
                    "as slow, small hardware microsteps. Never invent coordinates, "
                    "joint commands, settings, or claim hardware safety. Choose stop with target=null "
                    "if the task cannot be supported. State data after the initial capture are predicted "
                    "simulation states, not new hardware observations. A deterministic reverse path "
                    "will be validated before any physical motion. Return the JSON schema only."
                )},
                {"role": "user", "content": json.dumps({
                    **context, "candidates": [c.to_prompt() for c in candidates],
                    "previous_rejection": correction,
                }, allow_nan=False)},
            ],
            "max_tokens": 4096,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "nero_next_pose", "strict": True,
                "schema": response_schema([c.id for c in candidates]),
            }},
            "provider": {"require_parameters": True, "allow_fallbacks": False},
        }
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload, allow_nan=False).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                body = response.read(1_000_001)
        except urllib.error.HTTPError as error:
            # Avoid echoing provider bodies or authentication headers into logs.
            raise RuntimeError(f"OpenRouter HTTP {error.code}; check model ID, credentials, credits and schema support") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError) as error:
            detail = "timed out" if isinstance(error, (socket.timeout, TimeoutError)) else "failed"
            raise RuntimeError(f"OpenRouter request {detail} after {self.timeout:g}s; no motion approved") from None
        if len(body) > 1_000_000:
            raise PlanningError("OpenRouter response exceeded size limit")
        try:
            envelope = strict_json(body)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise PlanningError("OpenRouter refused or returned an incomplete response")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise PlanningError("OpenRouter did not return text JSON")
            return strict_json(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeError):
            raise PlanningError("Malformed OpenRouter response") from None


class OfflineChooser:
    """Deterministic test double, explicitly selected with --offline-demo."""

    model = "offline-test-double"

    def choose(self, context, candidates, correction=None):
        candidate = candidates[0]
        pose = asdict(candidate.trajectory.source_target_pose)
        pose.pop("reason")
        return {"candidate_id": candidate.id, "target": pose,
                "reason": "Offline plumbing test; no LLM request was made"}
