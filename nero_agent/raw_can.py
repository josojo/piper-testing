"""Receive-only Linux SocketCAN trace, isolated from the hardware stop path."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import time

# Linux socket ABI (64-bit ROS Humble hosts). No SDK connection or CAN writes.
SO_TIMESTAMPNS = 35
SO_RXQ_OVFL = 40


def decode_frame(data, ancillary, flags, wall_ns, monotonic_ns):
    if len(data) != 16 or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise RuntimeError('Truncated or unsupported CAN frame/timestamp')
    can_id, length, payload = struct.unpack('=IB3x8s', data)
    stamp = None
    drops = None
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
            seconds, nanos = struct.unpack('@ll', value)
            stamp = seconds * 1_000_000_000 + nanos
        if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
            drops = struct.unpack('=I', value)[0]
    if stamp is None or length > 8:
        raise RuntimeError('Missing kernel timestamp or invalid classic CAN length')
    return {'type': 'frame', 'kernel_unix_ns': stamp,
            'read_wall_ns': wall_ns, 'read_monotonic_ns': monotonic_ns,
            'can_id_with_flags': can_id, 'data_hex': payload[:length].hex(),
            'socket_drops_total': drops}


def record(output, channel, duration=1800., max_bytes=256*1024*1024):
    if struct.calcsize('@ll') != 16:
        raise RuntimeError('Raw capture currently requires a 64-bit Linux timestamp ABI')
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # Exclusive creation protects previous traces and configuration files.
    with Path(output).open('x') as stream, socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        bus.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        bus.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        bus.bind((channel,))
        written = 0

        def emit(row):
            nonlocal written
            line = json.dumps(row, separators=(',', ':'))+'\n'
            stream.write(line)
            written += len(line.encode())

        emit({'type': 'header', 'schema': 1, 'channel': channel,
              'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
              'receive_buffer_bytes': bus.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
              'duration_limit_s': duration, 'byte_limit': max_bytes,
              'scope': 'Passive classic CAN; kernel receive time, not motor acquisition time. '
                       'Drop counter covers this socket only; missing footer means incomplete capture.'})
        stream.flush()
        print('READY', flush=True)
        end = time.monotonic()+duration
        count = 0
        reason = 'stopped'
        try:
            while running:
                if time.monotonic() >= end or written >= max_bytes:
                    reason = 'capture_limit'
                    break
                if not select.select([bus], [], [], .1)[0]:
                    continue
                data, ancillary, flags, _ = bus.recvmsg(16, 128)
                emit(decode_frame(data, ancillary, flags, time.time_ns(), time.monotonic_ns()))
                count += 1
        except Exception as error:
            reason = 'error: '+str(error)
            raise
        finally:
            emit({'type': 'footer', 'reason': reason, 'frames': count,
                  'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()})
    if reason != 'stopped':
        print('Raw CAN recording ended: '+reason, file=sys.stderr, flush=True)


def start_capture(output):
    process = subprocess.Popen(
        [sys.executable, '-m', 'nero_agent.raw_can', '--output', str(output),
         '--channel', os.environ.get('NERO_CAN_CHANNEL', 'can0')],
        stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        if not select.select([process.stdout], [], [], 5)[0] or process.stdout.readline().strip() != 'READY':
            raise RuntimeError('Raw CAN recorder did not become ready; hardware stack not started')
    except BaseException:
        stop_capture(process)
        raise
    print('Passive raw CAN recording: '+str(output), flush=True)
    return process


def stop_capture(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
            print('Raw CAN recorder required forced shutdown; trace may be incomplete', file=sys.stderr)
    if process.returncode:
        print('Raw CAN recorder failed; inspect stderr and trace footer', file=sys.stderr)
    process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel', default=os.environ.get('NERO_CAN_CHANNEL', 'can0'))
    args = parser.parse_args()
    record(args.output, args.channel)


if __name__ == '__main__':
    main()
