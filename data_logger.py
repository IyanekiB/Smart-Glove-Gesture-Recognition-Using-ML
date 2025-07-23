import asyncio
from bleak import BleakClient, BleakScanner
import csv
import os

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

label = input("Enter gesture label (e.g. swipe_left): ")
sample_num = input("Enter sample number: ")
os.makedirs("data", exist_ok=True)
filename = os.path.join("data", f"{label}_{sample_num}.csv")

async def run(address):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        print("Connected. Logging BLE data. Ctrl+C to stop.")

        def handle_data(sender, data):
            try:
                line = data.decode('utf-8').strip()
                if line:
                    row = line.split(',')
                    writer.writerow(row)
            except Exception as e:
                print("Error:", e)

        async with BleakClient(address) as client:
            await client.start_notify(UART_TX_CHAR_UUID, handle_data)
            print("Receiving BLE notifications...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await client.stop_notify(UART_TX_CHAR_UUID)
                print("Stopped.")

# Discover BLE device address
async def discover_and_run():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)
        if d.name and "Nano33BLE-Gesture" in d.name:
            print("Found device:", d)
            await run(d.address)
            break
    else:
        print("Nano33BLE-Gesture not found. Make sure it is advertising!")

asyncio.run(discover_and_run())
