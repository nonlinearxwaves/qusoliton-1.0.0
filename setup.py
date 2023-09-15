#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="qusoliton",
    version="1.0",
    description="Quantum Soliton toolbox",
    url="https://github.com/nonlinearxwaves/qusoliton-1.0.0",
    author="Claudio Conti",
    author_email="nonlinearxwaves@gmail.com",
    packages=find_packages(),  # same as name
    # external packages as dependencies
    # install_requires=['wheel', 'bar', 'numpy', 'cupy'],
    # scripts=['qusoliton']
)
