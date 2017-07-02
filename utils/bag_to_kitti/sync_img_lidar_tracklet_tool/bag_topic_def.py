""" ROS Bag file topic definitions
"""

SINGLE_CAMERA_TOPIC = "/image_raw"
CAMERA_TOPICS = [SINGLE_CAMERA_TOPIC]

CAP_REAR_GPS_TOPICS = ["/capture_vehicle/rear/gps/fix", "/objects/capture_vehicle/rear/gps/fix"]
CAP_REAR_RTK_TOPICS = ["/capture_vehicle/rear/gps/rtkfix", "/objects/capture_vehicle/rear/gps/rtkfix"]
CAP_FRONT_GPS_TOPICS = ["/capture_vehicle/front/gps/fix", "/objects/capture_vehicle/front/gps/fix"]
CAP_FRONT_RTK_TOPICS = ["/capture_vehicle/front/gps/rtkfix", "/objects/capture_vehicle/front/gps/rtkfix"]

OBJECTS_TOPIC_ROOT = "/objects"

TF_TOPIC = "/tf"
