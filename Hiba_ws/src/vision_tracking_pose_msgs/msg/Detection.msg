# defines a YOLO detection result
# class probability
int32 class_id
string class_name
float64 score
# ID for tracking
string id
# 2D bounding box surrounding the object in pixels
vision_tracking_pose_msgs/BoundingBox2D bbox
# 3D bounding box surrounding the object in meters
vision_tracking_pose_msgs/BoundingBox3D bbox3d
# keypoints for human pose estimation
vision_tracking_pose_msgs/KeyPoint2DArray keypoints
# keypoints for human pose estimation
vision_tracking_pose_msgs/KeyPoint3DArray keypoints3d
