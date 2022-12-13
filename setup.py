# -*- coding: utf-8 -*-
"""
@author: Marten Veldmann
"""

from setuptools import setup

setup(name='dsv2pulseq',
      version="1.0",
      description="Create Pulseq sequences from dsv.",
      long_description=open('README.md').read(),
      keywords='pulseq,dsv,siemens,mri',
      author='Marten Veldmann',
      author_email='marten.veldmann@dzne.de',
      url='https://github.com/mavel101/dsv2pulseq',
      license='MIT License',
      packages=['dsv2pulseq'],
      scripts=['dsv2pulseq/dsv_to_pulseq.py'],
      dependencies = ['numpy, pypulseq'],
      zip_safe=False,
      test_suite="test",
      )
