from glob import glob
from os.path import join

from setuptools import setup


package_name = 'policy_action_controller'


setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    package_dir={'': 'script'},
    py_modules=[
        'backend_factory',
        'control_config',
        'lowcmd_controller',
        'motion_mode_manager',
        'observation_builder',
        'policy_runner',
        'policy_scheduler',
        'scheduler_runtime',
        'sim2real_backend',
        'sim2sim_backend',
    ],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [join('resource', package_name)],
        ),
        (join('share', package_name), ['package.xml']),
        (join('share', package_name, 'config'), glob('config/*')),
        (join('share', package_name, 'model'), glob('model/*')),
    ],
    scripts=[
        join('script', 'motion_mode_manager.py'),
        join('script', 'policy_scheduler.py'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ccwss',
    maintainer_email='ccwss@example.com',
    description='Policy action controller tools for Unitree Go2.',
    license='Apache-2.0',
)
