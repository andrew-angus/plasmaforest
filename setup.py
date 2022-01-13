#!/bin/python3

from setuptools import setup,find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name="plasmaforest",
  version="0.1.0",

  author="Andrew Angus",
  author_email="andrew.angus@warwick.ac.uk",

  packages=find_packages(include=['plasmaforest','plasmaforest.*']),

  url="https://github.com/andrewanguswarwick/plasmaforest",

  description="Laser-plasma quantity solver and handler",
  long_description=long_description,
  long_description_content_type="text/markdown",

  python_requires='>=3.6',
  install_requires=[
    "plasmapy",
    "numpy",
    "scipy"
    ],
)
