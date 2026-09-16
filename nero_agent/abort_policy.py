"""Prevent motion until a powered controlled-stop primitive is verified.

This is deliberately not a configurable acknowledgement: reviewing a scene
does not establish controller stopping behavior. Do not replace this guard
with move_js(current_position), a reset, or a disabled velocity check.
"""
import json
from pathlib import Path
from .core import AgentError


def require_verified_controlled_abort(report_path=None):
    if report_path is None:
        raise AgentError('Hardware execution unavailable until the powered controlled abort is verified; supply --abort-qualification-report pointing to a passed moving-abort report')
    path = Path(report_path)
    try:
        report = json.loads(path.read_text())
        commissioning = report['commissioning']
        abort = report['controlled_abort']
        metrics = commissioning['metrics']
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AgentError('Invalid abort qualification report: ' + str(error))
    if (report.get('status') != 'passed' or commissioning.get('status') != 'passed'
            or abort.get('status') != 'holding' or not metrics.get('standstill_dwell_confirmed')
            or metrics.get('time_to_first_standstill_s') is None):
        raise AgentError('Abort qualification report is not a passed, sustained-hold trial')
    return path
