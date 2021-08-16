import os
from setuptools import find_packages, setup

__version__ = None


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="DimeReaction",
    version=__version__,
    author="Kevin Spiekermann and Lagnajit Pattanaik",
    description="Uses a network inspired by dimenet++ to prediction reaction properties.",
    url="https://github.com/kspieks/DimeReaction",
    packages=find_packages(),
    long_description=read('README.md'),
)
