import can

bus = can.Bus(
      interface="gs_usb",
      channel=0,
      bitrate=1_000_000,
  )

print("CandleLight opened successfully")
bus.shutdown()
