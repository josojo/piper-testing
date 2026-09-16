#!/usr/bin/env bash
# Docker isolates ROS Humble from the host Python/Ubuntu installation.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image_name=nero-moveit:humble
operation=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$operation" in
  build)
    exec docker build --tag "$image_name" --file "$repo_root/ros2/Dockerfile" "$repo_root/ros2"
    ;;
  demo|hardware|capture)
    if ! docker info >/dev/null 2>&1; then
      echo 'Docker is unavailable or access is denied. Run this script with sudo, or use a ROS Humble environment directly. See README.' >&2
      exit 2
    fi
    if [[ ! -d "$repo_root/reports" && -n ${SUDO_UID:-} ]]; then
      install -d -o "$SUDO_UID" -g "$SUDO_GID" "$repo_root/reports"
    else
      mkdir -p "$repo_root/reports"
    fi
    runtime=(--rm --init --user "${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}" 
      --mount "type=bind,source=$repo_root,target=/work" --workdir /work
      --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-73}" --env ROS_LOCALHOST_ONLY=1)
    if [[ -t 0 && -t 1 ]]; then runtime+=(-it); fi
    if [[ -n ${OPENROUTER_API_KEY:-} ]]; then runtime+=(--env OPENROUTER_API_KEY); fi
    if [[ -n ${OPENROUTER_MODEL:-} ]]; then runtime+=(--env OPENROUTER_MODEL); fi
    if [[ "$operation" == demo ]]; then
      # Private network namespace has no host can0. No --privileged or device mounts.
      exec docker run "${runtime[@]}" "$image_name" python3 -m nero_agent.bringup \
        --config examples/nero-agent.mock.json --execute "$@"
    fi
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 $operation path/to/reviewed-config.json [--scripted] [--execute] [--output reports/result.json]" >&2
      exit 2
    fi
    config_path=$1
    shift
    # The configuration must be under this repository so it is visible in /work.
    config_path=$(realpath -- "$config_path")
    case "$config_path" in "$repo_root/"*) ;; *) echo 'Config must be inside the repository' >&2; exit 2;; esac
    config_path="/work/${config_path#"$repo_root/"}"
    runtime+=(--network host --env "NERO_CAN_CHANNEL=${NERO_CAN_CHANNEL:-can0}")
    if [[ "$operation" == capture ]]; then
      exec docker run "${runtime[@]}" "$image_name" python3 -m nero_agent.bringup \
        --config "$config_path" --capture-only "$@"
    fi
    exec docker run "${runtime[@]}" "$image_name" python3 -m nero_agent.bringup --config "$config_path" "$@"
    ;;
  *)
    echo "Usage: $0 build | demo [--scripted | --instruction TEXT] | capture CONFIG | hardware CONFIG [--scripted] [--execute]"
    ;;
esac
