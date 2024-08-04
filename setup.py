from setuptools import setup
from setuptools import find_packages

setup(
    name='shipment',
    version='0.0.1',
    author='VibhavUcharia',
    author_email='vibhavucharia.learn@gmail.com',
    package_dir={"": "shipment"},
    packages=find_packages(where="shipment")
)