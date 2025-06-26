#!/usr/bin/env python3
import socket
import json
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker

class UDPToRosBridge(Node):
    def __init__(self):
        super().__init__('udp_to_ros_bridge')
        
        # Publishers
        self.people_pub     = self.create_publisher(PoseArray, '/people_tracked', 10)
        self.human_in_map   = self.create_publisher(Marker,    '/human_markers',   10)
        
        # UDP setup
        self.udp_port = 12345
        self.sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.udp_port))
        self.sock.settimeout(1.0)
        
        self.previous_ids = []
        self.running      = True
        
        self.udp_thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.udp_thread.start()
        
        self.get_logger().info(f"UDP-to-ROS bridge listening on port {self.udp_port}")

    def udp_listener(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                msg_data   = json.loads(data.decode('utf-8'))
                self.get_logger().info(f"Received {len(msg_data['people'])} people from {addr}")
                
                self.publish_pose_array(msg_data)
                self.update_people_tracked(msg_data)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f"UDP receive error: {e}")
                time.sleep(0.1)

    def publish_pose_array(self, msg_data):
        pa = PoseArray()
        pa.header.frame_id = msg_data.get('frame_id','map')
        pa.header.stamp    = self.get_clock().now().to_msg()
        for person in msg_data['people']:
            p = Pose()
            p.position.x = float(person['x'])
            p.position.y = float(person['y'])
            p.position.z = float(person.get('z',0.0))
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.people_pub.publish(pa)

    def update_people_tracked(self, msg_data):
        people = msg_data['people']
        # delete if none
        if not people:
            for old_id in self.previous_ids:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns = 'humans'; m.id = old_id; m.action = Marker.DELETE
                self.human_in_map.publish(m)
            self.previous_ids = []
            return
        
        new_ids = []
        for idx, person in enumerate(people, start=1):
            new_ids.append(idx)
            # publish PoseArray done above
            
            # sphere marker
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'humans'; m.id = idx
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(person['x'])
            m.pose.position.y = float(person['y'])
            m.pose.position.z = float(person.get('z',0.0))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            
            # blue if teleco, else red
            cls = person.get('class','person')
            if cls.lower() == 'teleco':
                m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; m.color.a = 0.8
            else:
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 0.8

            self.human_in_map.publish(m)
            self.get_logger().info(f"Marker id={idx} ({cls}) @ x={person['x']:.2f}, y={person['y']:.2f}")
        
        # delete disappeared
        for old_id in self.previous_ids:
            if old_id not in new_ids:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns = 'humans'; m.id = old_id; m.action = Marker.DELETE
                self.human_in_map.publish(m)
        
        self.previous_ids = new_ids

    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main():
    rclpy.init()
    bridge = UDPToRosBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
