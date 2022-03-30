#!/usr/bin/env python
from distutils.core import setup

setup(name='psychXRF',
    version='0.1',
    description='Predict synthetic multichannel XRF spectra simulation parameters',
    author='Rosario Andolina',
    author_email='andolinarosario@gmail.com',
    packages=['psychXRF'],
    package_dir={'psychXRF' : 'src'}
    )
