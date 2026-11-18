from setuptools import setup

package_name = 'e50_runtime_bridge'
setup(name=package_name, version='0.1.0', packages=[package_name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                  ('share/' + package_name, ['package.xml'])],
      entry_points={'console_scripts': ['e50_runtime_bridge = e50_runtime_bridge.node:main']})
