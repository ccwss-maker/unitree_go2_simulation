from setuptools import setup

package_name = 'obs_preprocess'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'obs_preprocess_node = obs_preprocess.obs_node:main',
        ],
    },
    maintainer='ccwss',
    maintainer_email='ccwss@ccwss.com',
    description='Observation pre-processor for Go2 Isaac Sim to ROS2 deployment',
    license='Apache-2.0',
)
