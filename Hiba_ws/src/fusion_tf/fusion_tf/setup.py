from setuptools import setup, find_packages
from glob import glob

package_name = 'fusion_tf'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Fusion package including Python laser scan to point cloud converter.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan_to_cloud = fusion_tf.scan_to_cloud:main',
            'fusion = fusion_tf.fusion:main',
            'z_inspector = fusion_tf.z_inspector:main',
            'fusion_method1 = fusion_tf.fusion_method1:main',
        ],
    },
)