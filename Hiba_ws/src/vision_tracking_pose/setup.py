from setuptools import setup, find_packages

package_name = 'vision_tracking_pose'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Include any configuration files if needed
        ('share/' + package_name + '/config', [
            'vision_tracking_pose/bytetrack.yaml',
            'vision_tracking_pose/botsort.yaml'
        ])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Vision pipeline with YOLOv8 detection, tracking, 3D pose estimation and debug visualization for ROS 2',
    license='GPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'distance = vision_tracking_pose.distance:main',
            'visualization = vision_tracking_pose.visualization:main',
            'detection = vision_tracking_pose.detection:main',
            'dete_track_sim = vision_tracking_pose.dete_track_sim:main',
            'detec_track_hard = vision_tracking_pose.detec_track_hard:main',
            'pose_estimation_depth = vision_tracking_pose.pose_estimation_depth:main',
            'tracking_velocity = vision_tracking_pose.tracking_velocity:main',
            'detection_hardware = vision_tracking_pose.detection_hardware:main',
        ],
    },
)

