import socket
import struct
import unittest

from nero_agent.raw_can import SO_RXQ_OVFL, SO_TIMESTAMPNS, decode_frame, record


class RawCanTests(unittest.TestCase):
    def test_preserves_bytes_timestamp_flags_and_drop_counter(self):
        data = struct.pack('=IB3x8s', 0x253, 8, bytes.fromhex('ffdd010203040506'))
        ancillary = [(socket.SOL_SOCKET, SO_TIMESTAMPNS, struct.pack('@ll', 123, 456)),
                     (socket.SOL_SOCKET, SO_RXQ_OVFL, struct.pack('=I', 7))]
        row = decode_frame(data, ancillary, 0, 999, 888)
        self.assertEqual(row['kernel_unix_ns'], 123000000456)
        self.assertEqual(row['data_hex'], 'ffdd010203040506')
        self.assertEqual(row['can_id_with_flags'], 0x253)
        self.assertEqual(row['socket_drops_total'], 7)
        self.assertEqual(row['read_monotonic_ns'], 888)

    def test_rejects_missing_or_truncated_timestamp(self):
        frame = struct.pack('=IB3x8s', 0x253, 8, b'12345678')
        for data, flags in ((frame, 0), (frame, socket.MSG_CTRUNC), (b'', 0)):
            with self.assertRaises(RuntimeError):
                decode_frame(data, [], flags, 0, 0)

    def test_does_not_overwrite_existing_file(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            path.touch()
            with patch('nero_agent.raw_can.signal.signal'), patch('nero_agent.raw_can.socket.socket') as bus:
                with self.assertRaises(FileExistsError):
                    record(path, 'can0')
                bus.assert_not_called()

    def test_bounded_capture_has_footer_and_never_sends(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            with patch('nero_agent.raw_can.signal.signal'), patch('nero_agent.raw_can.socket.socket') as factory:
                bus = factory.return_value.__enter__.return_value
                bus.getsockopt.return_value = 2048
                with patch('builtins.print'):
                    record(path, 'can0', duration=0)
                bus.bind.assert_called_once_with(('can0',))
                bus.send.assert_not_called()
                bus.sendto.assert_not_called()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[0]['type'], 'header')
                self.assertEqual(rows[-1]['reason'], 'capture_limit')


if __name__ == '__main__':
    unittest.main()
