from setuptools import find_packages, setup

package_name = 'my_nmpc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eya',
    maintainer_email='eya@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
          'obstacle_debug_node = my_nmpc_controller.obstacle_debug_node:main',
          'distance_check_node = my_nmpc_controller.distance_check_node:main',
          'nmpc_node = my_nmpc_controller.nmpc_node:main',
          'people_bridge = my_nmpc_controller.people_bridge:main',
        ],
    },
)
