""" Setup file for QuSoliton : Quantum Soliton toolbox"""
#! /usr/bin/env python
import sys
from pathlib import Path
from setuptools import setup, find_packages

VERSION = '1.0.0'
THIS_DIR = Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIR / "README.md").read_text()

# defauls packages
PACKAGES = find_packages(include=['qusoliton',
                                  'qusoliton.sdenls',
                                  'qusoliton.cusdenls',
                                  ])

# external packages as dependencies
REQUIRES = ['wheel', 'numpy', 'termcolor',
            'cupy-cuda115', 'pillow', 'matplotlib',
            'hurry', 'hurry.filesize', ]


# check if compilation exclude cupy package
if '--no-cuda' in sys.argv:
    print("Installation withouth cupy-cuda")
    REQUIRES = ['wheel', 'numpy', 'termcolor',
                'pillow', 'matplotlib',
                'hurry', 'hurry.filesize', ]
# remove no-cuda option before running setup
if "--no-cuda" in sys.argv:
    sys.argv.remove("--no-cuda")


setup(
    name='qusoliton',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    version=VERSION,
    description='Quantum Soliton toolbox',
    author='Claudio Conti',
    author_email='nonlinearxwaves@gmail.com',
    #    packages=['qusoliton', 'qusoliton.sdenls',
    #              'qusoliton.cusdenls'],  # same as name
    #    packages=find_packages(include=['qusoliton', 'qusoliton.*'])
    packages=PACKAGES,

    # external packages as dependencies
    install_requires=REQUIRES,
)
