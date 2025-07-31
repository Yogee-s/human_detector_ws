#!/usr/bin/env python3
import json
import sys
import threading

import paho.mqtt.client as mqtt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MQTT_USERNAME = "commu"
MQTT_PASSWORD = "zD5%rZ$m/i+W"
MQTT_ADDRESS  = "mittsu-talk.jp"
MQTT_PORT     = 443
MQTT_PATH     = "/mosmos-test2/ws/"
MQTT_TOPIC    = "/topic/testing/testroom/ROVER-001/info"
# ───────────────────────────────────────────────────────────────────────────────

last_point = None
lock = threading.Lock()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[mqtt] connected, subscribing…")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[mqtt] connect failed (rc={rc})", file=sys.stderr)

def on_message(client, userdata, msg):
    global last_point
    raw = msg.payload.decode('utf-8', errors='ignore').strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # if it's not JSON, always print it
        print(f"[{msg.topic}] {raw}")
        return

    # try both 'teleco' and 'pose.position' styles
    coords = None
    if 'teleco' in data and isinstance(data['teleco'], dict):
        t = data['teleco']
        if all(k in t for k in ('x','y','z')):
            coords = (float(t['x']), float(t['y']), float(t['z']))
    elif 'pose' in data and isinstance(data['pose'], dict):
        p = data['pose'].get('position', {})
        if all(k in p for k in ('x','y','z')):
            coords = (float(p['x']), float(p['y']), float(p['z']))

    with lock:
        if coords is None:
            # no coordinate in this message — ignore or print if you like
            return

        if coords != last_point:
            # only print when it changes
            print(f"[{msg.topic}] {raw}")
            last_point = coords

def main():
    client = mqtt.Client(client_id="debug", transport="websockets")
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.ws_set_options(path=MQTT_PATH)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_ADDRESS, MQTT_PORT)
    except Exception as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    client.loop_forever()

if __name__ == "__main__":
    main()
