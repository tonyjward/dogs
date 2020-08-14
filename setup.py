#!/usr/bin/env python

from setuptools import setup

setup(name='the-dogs',
      version='0.1',
      # list folders, not files
      packages=['the-dogs',
                'the-dogs.tests'],
      scripts=['the-dogs/bin/basic_script.py'],
      )
