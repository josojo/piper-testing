#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /opt/nero_ws/install/setup.bash
export HOME=/tmp/nero-home
mkdir -p "$HOME"
exec "$@"
