"""Plan-only worker for Viser, connected to an already running ROS stack."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .core import Settings, strict_json

ROOT = Path(__file__).resolve().parents[1]


def target_config(config, position, orientation):
    result = dict(config)
    result['named_poses'] = {'viewer_target': {
        'frame': 'base_link', 'position_m': list(position),
        'orientation_xyzw': list(orientation)}}
    settings = Settings.parse(result, require_review=False)
    if not settings.mujoco_preflight or settings.segmented_execution:
        raise ValueError('Viewer planning requires a continuous MuJoCo motion profile')
    return result


class ViewerPlanner:
    def __init__(self, config, container=None):
        self.config = Path(config).resolve()
        self.container = container
        self.defaults = strict_json(self.config.read_text())

    def __call__(self, position, orientation):
        config = target_config(self.defaults, position, orientation)
        directory = Path(tempfile.mkdtemp(prefix='viewer-plan-', dir=ROOT / 'reports'))
        request = directory / 'config.json'
        preview = directory / 'preview.json'
        request.write_text(json.dumps(config, allow_nan=False))
        def mapped(path):
            return str(Path('/work') / path.relative_to(ROOT)) if self.container else str(path)
        command = [sys.executable, '-m', 'nero_agent.viewer_planning',
                   '--config', mapped(request), '--output', mapped(preview)]
        if self.container:
            command[0] = 'python3'
            command = ['docker', 'exec', '-w', '/work', self.container, '/nero_entrypoint.sh'] + command
        with (directory / 'planner.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=240, check=False)
        if preview.exists():
            return preview  # Rejected candidates are also useful, explicitly labelled.
        detail = (directory / 'planner.log').read_text()[-2500:]
        raise RuntimeError('Planning failed (exit %d): %s' % (result.returncode, detail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from .ros_backend import RosBackend
    settings = Settings.parse(strict_json(args.config.read_text()), require_review=False)
    backend = RosBackend(settings)
    try:
        backend.preview_output = args.output
        state = backend.state()
        if max(map(abs, state['velocities_rad_s'])) > .003:
            raise ValueError('Planning requires a stationary robot')
        result = backend.plan(settings.named_poses['viewer_target'])
        args.output.with_suffix('.plan.json').write_text(json.dumps(result, indent=2))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
