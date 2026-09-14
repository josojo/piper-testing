import can
import time
from nero_safety_common import nero_can_config

bus = can.Bus(**nero_can_config())

print("Listening for CAN frames for 10 seconds...")

try:
      end = time.time() + 10
      count = 0

      while time.time() < end:
          message = bus.recv(timeout=1)

          if message is not None:
              count += 1
              print(message)

      print(f"Received {count} CAN frames")

finally:
      bus.shutdown()
