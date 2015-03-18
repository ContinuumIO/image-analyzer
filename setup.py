#!/usr/bin/env python

import os
from setuptools import setup

setup(name='image_analyzer',
      version='0.0.0',
      description='Data migration utilities',
      url='http://github.com/ContinuumIO/image-analyzer/',
      author='Peter Steinberg',
      author_email='psteinberg@continuum.io',
      license='BSD',
      keywords='image similarity analysis',
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      long_description=(open('README.rst').read() if os.path.exists('README.rst')
                        else ''),
      modules=['image_mapper','on_each_image','similarity'],
      zip_safe=False)
