#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='qusoliton',
    version='1.0',
    description='Quantum Soliton toolbox',
    author='Claudio Conti',
    author_email='nonlinearxwaves@gmail.com',
    packages=['qusoliton', 'qusoliton.sdenls',
              'qusoliton.cusdenls'],  # same as name
    #  packages=find_packages(include=['exampleproject', 'exampleproject.*'])

    # external packages as dependencies
    #   install_requires=['wheel', 'bar', 'numpy', 'termcolor', 'cupy'],
)
