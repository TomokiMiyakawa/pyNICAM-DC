#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from setuptools.extension import Extension
from codecs import open
import sys
import os


here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
import versioneer  # noqa: E402


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

INSTALL_REQUIRES = [
    "numpy>=1.24.0"
]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="pynicamdc",
    license="MIT",
    author="Tomoki Miyakawa (University of Tokyo)",
    author_email="miyakawa@aori.u-tokyo.ac.jp",
    keywords="atmospheric dynamics python parallel numpy multi-core geophysics atmospheric-model mpi4py",
    description="The dynamical core of Nonhydrostatic ICosahedral Atmospheric Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url='https://pynicamdc.readthedocs.io',
    python_requires=">=3.9",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
)
