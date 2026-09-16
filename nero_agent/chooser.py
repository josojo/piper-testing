"""Structured LLM action selection; no ROS or hardware access."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .core import AgentError, strict_json


class OpenRouterChooser:
    def __init__(self, model=None, transport=urllib.request.urlopen):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        except ImportError:
            pass  # An exported API key needs no optional dotenv dependency.
        self.model = model or os.environ.get('OPENROUTER_MODEL', 'gpt-5.6-luna')
        self.key = os.environ.get('OPENROUTER_API_KEY')
        if not self.key:
            raise AgentError('Export OPENROUTER_API_KEY, or install python-dotenv to load .env')
        self.transport = transport

    def choose(self, context):
        schema = {'type': 'object', 'additionalProperties': False,
                  'required': ['action', 'pose', 'reason'], 'properties': {
                      'action': {'type': 'string', 'enum': ['get_state', 'move_to_named_pose', 'stop', 'finish']},
                      'pose': {'anyOf': [{'type': 'null'}, {'type': 'string', 'enum': list(context['named_poses'])}]},
                      'reason': {'type': 'string'}}}
        payload = {'model': self.model, 'messages': [
            {'role': 'system', 'content': (
                'Choose one action for a seven-joint NERO arm. Use only the supplied named poses. '
                'move_to_named_pose requires a pose name; other actions require pose=null. '
                'Inspect action results and fresh measured state before deciding the next action. '
                'Use finish only when the instruction is complete; use stop when unsupported. '
                'Never invent coordinates, settings, tools, or claim physical safety. '
                'The source field distinguishes physical feedback from mock and offline tests. '
                'Return only the required JSON object.')},
            {'role': 'user', 'content': json.dumps(context, allow_nan=False)}],
            'max_tokens': 1000, 'provider': {'require_parameters': True, 'allow_fallbacks': False},
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'nero_action', 'strict': True, 'schema': schema}}}
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(payload, allow_nan=False).encode(), method='POST',
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with self.transport(request, timeout=60) as response:
                body = response.read(1_000_001)
        except urllib.error.HTTPError as error:
            raise AgentError('OpenRouter HTTP %d; check model, credentials and structured-output support' % error.code) from None
        except (OSError, TimeoutError):
            raise AgentError('OpenRouter request failed or timed out') from None
        if len(body) > 1_000_000:
            raise AgentError('OpenRouter response too large')
        try:
            choice = strict_json(body)['choices'][0]
            if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
                raise AgentError('LLM refused or returned an incomplete action')
            return strict_json(choice['message']['content'])
        except (KeyError, IndexError, TypeError, ValueError):
            raise AgentError('Malformed LLM action response') from None
