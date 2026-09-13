import can
import time

bus = can.Bus(
      interface="gs_usb",
      channel=0,
      bitrate=1_000_000,
  )

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
