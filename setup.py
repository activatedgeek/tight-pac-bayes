import os
from setuptools import setup, find_packages


if os.path.isfile('VERSION'):
  with open('VERSION') as f:
    VERSION = f.read()
else:
  VERSION = '0.0.dev0'

with open('README.md') as f:
  README = f.read()

with open('requirements.txt') as f:
  requirements = f.read().split()

setup(name='pactl',
      description='Transfer Learning and PAC-Bayes',
      long_description=README,
      long_description_content_type='text/markdown',
      version=VERSION,
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(exclude=[
          'config',
          'config.*',
          'experiments',
          'experiments.*',
          'notebooks',
          'notebooks.*',
      ]),
      python_requires='>=3.6, <3.10',
      install_requires=requirements,
      extras_require={}
     )