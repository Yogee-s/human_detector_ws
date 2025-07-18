#!/usr/bin/env python3
# MQTT Subscriber and ROS2 bridge with config file support and debugging

import json
import yaml
import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker
import os
import sys

# ─── DEFAULT CONFIG (if no file provided) ─────────────────────────────────────
DEFAULT_CONFIG = {
    'MQTTUSERNAME': 'commu',
    'MQTTPASSWORD': 'zD5%rZ$m/i+W',
    'MQTTADDRESS': 'mittsu-talk.jp',
    'MQTTPATH': '/mosmos-test2/ws/',
    'TOPICS': {
        'HUMAN_DETECTION': '/topic/testing/testroom/realsense_human_detection'
    }
}

# Toggle these to show/hide each class
SHOW_TELECO = False
SHOW_PERSON = True
# ───────────────────────────────────────────────────────────────────────────────

class MqttHumanBridge(Node):
    def __init__(self, config_file=None):
        super().__init__('mqtt_human_bridge')

        # Load configuration
        self.config = self.load_config(config_file)
        
        # ROS publishers
        self.people_pub = self.create_publisher(PoseArray, '/people_tracked', 10)
        self.marker_pub = self.create_publisher(Marker,    '/human_markers',   10)

        self._prev_count = 0
        self._message_count = 0

        # MQTT client setup
        self.setup_mqtt()

        self.get_logger().info("MQTT Human Bridge initialized")
        self.get_logger().info(f"Config: {self.config}")

    def load_config(self, config_file):
        """Load configuration from YAML file or use defaults"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.get_logger().info(f"Loaded config from {config_file}")
                return config
            except Exception as e:
                self.get_logger().error(f"Failed to load config file: {e}")
                return DEFAULT_CONFIG
        else:
            self.get_logger().warn("No config file provided or file not found, using defaults")
            return DEFAULT_CONFIG

    def setup_mqtt(self):
        """Setup MQTT client with configuration"""
        self._mqtt = mqtt.Client(client_id="ros2_bridge", transport="websockets")
        
        # Enable TLS
        self._mqtt.tls_set()
        
        # Set credentials
        self._mqtt.username_pw_set(
            self.config['MQTTUSERNAME'], 
            self.config['MQTTPASSWORD']
        )
        
        # Set WebSocket options
        self._mqtt.ws_set_options(path=self.config['MQTTPATH'])
        
        # Set callbacks
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_message = self._on_message
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_log = self._on_log
        
        # Connect
        try:
            self.get_logger().info(f"Connecting to MQTT broker at {self.config['MQTTADDRESS']}:443")
            self._mqtt.connect(self.config['MQTTADDRESS'], 443)
            self._mqtt.loop_start()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to MQTT: {e}")

    def _on_log(self, client, userdata, level, buf):
        """MQTT logging callback"""
        self.get_logger().debug(f"MQTT Log: {buf}")

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            topic = self.config['TOPICS']['HUMAN_DETECTION']
            client.subscribe(topic)
            self.get_logger().info(f"✓ Connected to MQTT and subscribed to {topic}")
        else:
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            error_msg = error_messages.get(rc, f"Unknown error code {rc}")
            self.get_logger().error(f"✗ MQTT connect failed: {error_msg}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        if rc != 0:
            self.get_logger().warn(f"Unexpected MQTT disconnection (code: {rc})")
        else:
            self.get_logger().info("MQTT disconnected cleanly")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        self._message_count += 1
        
        try:
            # Log raw message for debugging
            raw_msg = msg.payload.decode('utf-8')
            self.get_logger().info(f"Message #{self._message_count} received on topic: {msg.topic}")
            self.get_logger().debug(f"Raw MQTT message: {raw_msg}")
            
            data = json.loads(raw_msg)
            self.get_logger().debug(f"Parsed JSON keys: {list(data.keys())}")
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON decode error: {e}")
            self.get_logger().error(f"Raw message: {raw_msg}")
            return
        except Exception as e:
            self.get_logger().error(f"Message processing error: {e}")
            return

        # Extract data - handle both formats
        teleco = data.get('teleco') if SHOW_TELECO else None
        people = data.get('people', []) if SHOW_PERSON else []
        
        self.get_logger().info(f"Processing: {len(people)} people, teleco: {'yes' if teleco else 'no'}")

        # Debug: Print people data structure
        for i, person in enumerate(people):
            self.get_logger().debug(f"Person {i}: {person}")

        # 1) Publish PoseArray
        pa = PoseArray()
        pa.header.frame_id = data.get('frame_id', 'map')
        pa.header.stamp = self.get_clock().now().to_msg()
        count = 0

        if teleco:
            p = Pose()
            p.position.x = float(teleco.get('x', 0.0))
            p.position.y = float(teleco.get('y', 0.0))
            p.position.z = float(teleco.get('z', 0.0))
            p.orientation.w = 1.0
            pa.poses.append(p)
            count += 1
            self.get_logger().debug(f"Added teleco at ({p.position.x:.2f}, {p.position.y:.2f}, {p.position.z:.2f})")

        for person in people:
            p = Pose()
            p.position.x = float(person.get('x', 0.0))
            p.position.y = float(person.get('y', 0.0))
            p.position.z = float(person.get('z', 0.0))
            p.orientation.w = 1.0
            pa.poses.append(p)
            count += 1
            self.get_logger().debug(f"Added person at ({p.position.x:.2f}, {p.position.y:.2f}, {p.position.z:.2f})")

        if count > 0:
            self.people_pub.publish(pa)
            self.get_logger().info(f"✓ Published {count} poses to /people_tracked")

        # 2) Publish Markers
        marker_id = 1

        if teleco:
            m = self.create_marker(teleco, marker_id, data.get('frame_id', 'map'), 'teleco')
            self.marker_pub.publish(m)
            marker_id += 1

        for person in people:
            m = self.create_marker(person, marker_id, data.get('frame_id', 'map'), 'person')
            self.marker_pub.publish(m)
            marker_id += 1

        if count > 0:
            self.get_logger().info(f"✓ Published {count} markers to /human_markers")

        # 3) Delete any old markers
        deleted_count = 0
        for old_id in range(marker_id, self._prev_count + 1):
            m = Marker()
            m.header.frame_id = data.get('frame_id', 'map')
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.action = 'humans', old_id, Marker.DELETE
            self.marker_pub.publish(m)
            deleted_count += 1

        if deleted_count > 0:
            self.get_logger().debug(f"Deleted {deleted_count} old markers")

        self._prev_count = marker_id - 1

    def create_marker(self, data, marker_id, frame_id, marker_type):
        """Create a marker from position data"""
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'humans', marker_id, Marker.SPHERE, Marker.ADD
        m.pose.position.x = float(data.get('x', 0.0))
        m.pose.position.y = float(data.get('y', 0.0))
        m.pose.position.z = float(data.get('z', 0.0))
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        
        # Color based on type
        if marker_type == 'teleco':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 1.0  # Blue
        else:  # person
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0  # Red
        
        return m

    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Shutting down MQTT bridge...")
        self._mqtt.loop_stop()
        self._mqtt.disconnect()
        super().destroy_node()


def main():
    rclpy.init()
    
    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    node = MqttHumanBridge(config_file)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()