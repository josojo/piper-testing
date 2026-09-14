import can
import os
from nero_safety_common import nero_can_config

bus = can.Bus(**nero_can_config())

print(
      "macOS CandleLight adapter opened successfully "
      f"({os.environ.get('NERO_CAN_INTERFACE', 'gs_usb')})"
)
bus.shutdown()
