from setuptools import setup
import os
from glob import glob

package_name = 'gz_sim_launcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        *[(os.path.join('share', package_name, 'models', os.path.relpath(root, 'models')),
           [os.path.join(root, f) for f in files])
          for root, dirs, files in os.walk('models')],
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ccwss',
    maintainer_email='ccwss',
    description='Launch Gz Sim with moon world',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'odom_to_tf = gz_sim_launcher.odom_to_tf:main',
            'foot_contact_processor = gz_sim_launcher.foot_contact_processor:main',
            'force_visualizer = gz_sim_launcher.force_visualizer:main',
            'unitree_legged_control = gz_sim_launcher.unitree_legged_control:main',
        ],
    },
)