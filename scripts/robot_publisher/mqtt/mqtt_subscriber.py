# MQTT Subsciber and ros2 bridge
import json
import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker

MQTT_USERNAME = "commu"
MQTT_PASSWORD = "zD5%rZ$m/i+W"
MQTT_ADDRESS  = "mittsu-talk.jp"
MQTT_PORT     = 443
MQTT_PATH     = "/mosmos-test2/ws/"
MQTT_TOPIC    = "/topic/testing/testroom/realsense_human_detection"

class MqttHumanBridge(Node):
    def __init__(self):
        super().__init__('mqtt_human_bridge')

        # ROS publishers
        self.people_pub = self.create_publisher(PoseArray, '/people_tracked', 10)
        self.marker_pub = self.create_publisher(Marker,    '/human_markers',   10)

        # track previous marker count for deletion
        self._prev_count = 0

        # MQTT client setup
        self._mqtt = mqtt.Client(client_id="ros2_bridge", transport="websockets")
        self._mqtt.tls_set()
        self._mqtt.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._mqtt.ws_set_options(path=MQTT_PATH)
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_message = self._on_message

        self._mqtt.connect(MQTT_ADDRESS, MQTT_PORT)
        self._mqtt.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
        else:
            self.get_logger().error(f"MQTT connect failed code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode('utf-8'))
        except json.JSONDecodeError:
            return

        # pull out teleco plus people
        teleco = data.get('teleco')
        people = data.get('people', [])

        # 1) Publish PoseArray
        pa = PoseArray()
        pa.header.frame_id = 'map'
        pa.header.stamp = self.get_clock().now().to_msg()
        count = 0

        if teleco:
            p = Pose()
            p.position.x = float(teleco.get('x',0.0))
            p.position.y = float(teleco.get('y',0.0))
            p.position.z = float(teleco.get('z',0.0))
            p.orientation.w = 1.0
            pa.poses.append(p)
            count += 1

        for person in people:
            p = Pose()
            p.position.x = float(person.get('x',0.0))
            p.position.y = float(person.get('y',0.0))
            p.position.z = float(person.get('z',0.0))
            p.orientation.w = 1.0
            pa.poses.append(p)
            count += 1

        self.people_pub.publish(pa)

        # 2) Publish Markers
        marker_id = 1

        if teleco:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns     = 'humans'
            m.id     = marker_id
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(teleco.get('x',0.0))
            m.pose.position.y = float(teleco.get('y',0.0))
            m.pose.position.z = float(teleco.get('z',0.0))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            # teleco = blue
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 1.0
            self.marker_pub.publish(m)
            marker_id += 1

        for person in people:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns     = 'humans'
            m.id     = marker_id
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(person.get('x',0.0))
            m.pose.position.y = float(person.get('y',0.0))
            m.pose.position.z = float(person.get('z',0.0))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            # person = red
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
            self.marker_pub.publish(m)
            marker_id += 1

        # 3) Delete any old markers
        for old_id in range(marker_id, self._prev_count+1):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns     = 'humans'
            m.id     = old_id
            m.action = Marker.DELETE
            self.marker_pub.publish(m)

        self._prev_count = marker_id - 1

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

if __name__=='__main__':
    main()
