#!/usr/bin/env python

from setuptools import setup

setup(name='dogs',
      version='0.1',
      # list folders, not files
      packages=['dogs',
                'dogs.tests'],
      scripts=['dogs/bin/basic_script.py'],
      )
