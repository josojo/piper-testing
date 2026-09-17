"""Structured LLM action selection; no ROS or hardware access."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .core import AgentError, strict_json


def decode_action_response(body):
    """Decode Chat Completions structured output without assuming one wire shape."""
    try:
        response = strict_json(body)
        choice = response['choices'][0]
        message = choice['message']
    except (KeyError, IndexError, TypeError, ValueError):
        raise AgentError('Malformed LLM action response: missing choices/message') from None
    if choice.get('finish_reason') != 'stop':
        raise AgentError('LLM refused or returned an incomplete action (finish_reason=%s)' %
                         choice.get('finish_reason', 'missing'))
    if message.get('refusal'):
        raise AgentError('LLM refused to return an action')

    # Providers normally return a JSON string. Some return the parsed object
    # directly, and some wrap text in the standard content-block array.
    content = message.get('content')
    if content is None and 'parsed' in message:
        content = message['parsed']
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if not isinstance(block, dict) or block.get('type') not in (None, 'text'):
                raise AgentError('Malformed LLM action response: unsupported content block')
            value = block.get('text')
            if not isinstance(value, str):
                raise AgentError('Malformed LLM action response: content block has no text')
            text_blocks.append(value)
        content = ''.join(text_blocks)
    if isinstance(content, dict):
        action = content
    elif isinstance(content, str):
        text = content.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
            if text.startswith('json'):
                text = text[4:].lstrip()
        try:
            action = strict_json(text)
        except (TypeError, ValueError):
            raise AgentError('Malformed LLM action response: content is not a JSON object') from None
    else:
        raise AgentError('Malformed LLM action response: unsupported content type')
    if not isinstance(action, dict):
        raise AgentError('Malformed LLM action response: action is not an object')
    return action


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
        return decode_action_response(body)
