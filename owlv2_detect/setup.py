from setuptools import find_packages, setup

package_name = 'owlv2_detect'

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
    maintainer='rusanrod',
    maintainer_email='rsandovr@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'owlv2_node = owlv2_detect.OWL_service:main',
            'Analize_node = owlv2_detect.Analize_node:main',
            'Analize_Human_node = owlv2_detect.Analize_Human_node:main',
            'owl_2d= owlv2_detect.owl_2d:main',
        ],
    },
)
