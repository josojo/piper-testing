#!/usr/bin/env bash
# Read measured joints and compute tcp_link pose in base_link. No execution gate.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -gt 2 ]]; then
  echo "Usage: sudo bash scripts/capture_nero_pose.sh [CONFIG] [OUTPUT]" >&2
  exit 2
fi
cd "$repo_root"
exec bash scripts/nero_ros2.sh capture "${1:-nero-agent.local.json}" \
  --tcp-pose --output "${2:-reports/current-tcp-pose.json}"
