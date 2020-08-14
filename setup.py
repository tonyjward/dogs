#!/usr/bin/env python

from setuptools import setup

setup(name='greyhound',
      version='0.1',
      # list folders, not files
      packages=['greyhound',
                'greyhound.tests'],
      scripts=['greyhound/bin/basic_script.py'],
      )