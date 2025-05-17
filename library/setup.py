from setuptools import find_packages, setup

setup(
    name='SaGE',
    packages=find_packages(),
    version='0.1.0',
    description='The Library to run SaGE',
    author='vnnm',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)