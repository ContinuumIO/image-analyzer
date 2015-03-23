#!/usr/bin/env python

import os
from setuptools import setup

setup(name='image_analyzer',
      version='0.1.1',
      description='Searching images using hashes and clustering',
      url='http://github.com/ContinuumIO/image-analyzer/',
      author='Peter Steinberg',
      author_email='psteinberg@continuum.io',
      license='BSD',
      keywords='image similarity analysis',
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      long_description=(open('README.rst').read() if os.path.exists('README.rst')
                        else ''),
      py_modules=['image_mapper',
                  'map_each_image',
                  'fuzzify_training', 
                  'load_data',
                  'search',
                  'hdfs_paths'],
      zip_safe=False)
