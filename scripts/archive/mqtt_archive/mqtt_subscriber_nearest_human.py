#!/usr/bin/env python3
# MQTT Subscriber and ROS2 bridge

import json
import threading

import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MQTT_USERNAME        = "commu"
MQTT_PASSWORD        = "zD5%rZ$m/i+W"
MQTT_ADDRESS         = "mittsu-talk.jp"
MQTT_PORT            = 443
MQTT_PATH            = "/mosmos-test2/ws/"
MQTT_TOPIC           = "/topic/testing/testroom/realsense_human_detection"

SHOW_DET_TELECO      = True     # show all YOLO-detected telecos
SHOW_PERSON_MARKERS  = True     # show all detected humans
HIGHLIGHT_NEAREST    = True     # highlight nearest human (green)
HIDE_OWN_TELECO      = True     # hide the detection that matches the robot itself
OWN_HIDE_DISTANCE    = 0.5      # meters threshold to consider “own” teleco
# ───────────────────────────────────────────────────────────────────────────────

class MqttHumanBridge(Node):
    def __init__(self):
        super().__init__('mqtt_human_bridge')

        # Publishers
        self.people_pub     = self.create_publisher(PoseArray, '/people_tracked', 10)
        self.marker_pub     = self.create_publisher(Marker,    '/human_markers',  10)

        # State
        self._lock          = threading.Lock()
        self._actual        = None   # (x,y) from /amcl_pose
        self._prev_cnt      = 0
        self.nearest_person = None   # (x, y, z) of closest human

        # AMCL pose subscription for true robot pose
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._on_amcl_pose,
            10
        )

        # MQTT setup
        self._mqtt = mqtt.Client(client_id="ros2_bridge", transport="websockets")
        self._mqtt.tls_set()
        self._mqtt.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._mqtt.ws_set_options(path=MQTT_PATH)
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_message = self._on_message
        self._mqtt.connect(MQTT_ADDRESS, MQTT_PORT)
        self._mqtt.loop_start()

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        with self._lock:
            self._actual = (p.x, p.y)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
            self.get_logger().info(f"[mqtt] connected → subscribed to {MQTT_TOPIC}")
        else:
            self.get_logger().error(f"[mqtt] connect failed rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode('utf-8'))
        except json.JSONDecodeError:
            return

        # pull teleco + people from payload
        teleco = data.get('teleco') if SHOW_DET_TELECO else None
        people = data.get('people', [])  if SHOW_PERSON_MARKERS else []

        # hide your own robot detection if configured
        with self._lock:
            actual = self._actual
        if teleco and HIDE_OWN_TELECO and actual:
            dx = teleco['x'] - actual[0]
            dy = teleco['y'] - actual[1]
            if dx*dx + dy*dy < OWN_HIDE_DISTANCE**2:
                teleco = None

        # 1) Publish all human poses as a PoseArray
        pa = PoseArray()
        pa.header.frame_id = 'map'
        pa.header.stamp    = self.get_clock().now().to_msg()
        for p in people:
            pose = Pose()
            pose.position.x = float(p['x'])
            pose.position.y = float(p['y'])
            pose.position.z = float(p['z'])
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        if pa.poses:
            self.people_pub.publish(pa)

        # 2) Compute nearest human (fallback to teleco if no /amcl_pose)
        ref_x, ref_y = None, None
        if HIGHLIGHT_NEAREST and people:
            if actual:
                ref_x, ref_y = actual
            elif teleco:
                ref_x, ref_y = teleco['x'], teleco['y']

            if ref_x is not None:
                dists = [
                    (i, (float(p['x'])-ref_x)**2 + (float(p['y'])-ref_y)**2)
                    for i,p in enumerate(people)
                ]
                nearest_idx, _ = min(dists, key=lambda kv: kv[1])
                np_ = people[nearest_idx]
                self.nearest_person = (
                    float(np_['x']), float(np_['y']), float(np_['z'])
                )
            else:
                nearest_idx = None
                self.nearest_person = None
        else:
            nearest_idx = None
            self.nearest_person = None

        # 3) Publish markers
        mid = 1

        # teleco detections (orange)
        if teleco:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id = 'teleco_det', mid
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x = float(teleco['x'])
            m.pose.position.y = float(teleco['y'])
            m.pose.position.z = float(teleco['z'])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            self.marker_pub.publish(m)
            mid += 1

        # human markers: nearest in green, rest in red
        for i, p in enumerate(people):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id = 'humans', mid
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x = float(p['x'])
            m.pose.position.y = float(p['y'])
            m.pose.position.z = float(p['z'])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3

            if HIGHLIGHT_NEAREST and i == nearest_idx:
                # nearest human in green
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            else:
                # standard human in red
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0

            self.marker_pub.publish(m)
            mid += 1

        # 4) Delete any old markers
        for old_id in range(mid, self._prev_cnt + 1):
            dm = Marker()
            dm.header.frame_id = 'map'
            dm.header.stamp    = self.get_clock().now().to_msg()
            dm.ns, dm.id, dm.action = 'humans', old_id, Marker.DELETE
            self.marker_pub.publish(dm)

        self._prev_cnt = mid - 1

    def destroy_node(self):
        self._mqtt.loop_stop()
        self._mqtt.disconnect()
        super().destroy_node()


def main():
    rclpy.init()
    node = MqttHumanBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
