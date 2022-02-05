""" Setup file for QuSoliton : Quantum Soliton toolbox"""
#! /usr/bin/env python
from pathlib import Path
from setuptools import setup, find_packages


THIS_DIR = Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIR / "README.md").read_text()

setup(
    name='qusoliton',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    version='1.0',
    description='Quantum Soliton toolbox',
    author='Claudio Conti',
    author_email='nonlinearxwaves@gmail.com',
    #    packages=['qusoliton', 'qusoliton.sdenls',
    #              'qusoliton.cusdenls'],  # same as name
    packages=find_packages(include=['qusoliton', 'qusoliton.*']),

    # external packages as dependencies
    install_requires=['wheel', 'numpy', 'termcolor',
                      'cupy-cuda115', 'pillow', 'matplotlib',
                      'hurry', 'hurry.filesize', ],
)
