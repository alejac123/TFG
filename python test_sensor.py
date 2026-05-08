import websocket
import json

def on_message(ws, message):
    print("MENSAJE:")
    print(message)
    print("---")

def on_error(ws, error):
    print(f"Error: {error}")

def on_open(ws):
    print("Conectado!")

ws = websocket.WebSocketApp(
    "wss://api.sensorcast.app/stream/alejandra",
    on_message=on_message,
    on_error=on_error,
    on_open=on_open,
    header={
        "Origin": "https://sensorcast.app",
        "User-Agent": "Mozilla/5.0"
    }
)

ws.run_forever()