#!/bin/python3

from setuptools import setup,find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name="plasmaforest",
  version="1.0.0",

  author="Andrew Angus",
  author_email="andrew.angus@warwick.ac.uk",

  packages=find_packages(include=['plasmaforest','plasmaforest.*']),

  url="https://github.com/andrewanguswarwick/plasmaforest",

  description="Plasma quantity solver and handler",
  long_description=long_description,
  long_description_content_type="text/markdown",

  python_requires='>=3.8',
  install_requires=[
    "plasmapy",
    "numpy",
    "scipy",
    "astropy",
    "typeguard",
    ],
)
