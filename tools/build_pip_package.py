# Copyright 2023 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module is used for building the Gematria python package. Usage:

bazel run //tools:build_pip_package bdist_wheel

This will build a Gematria python wheel that can then be used as a normal
python wheel.
"""

from datetime import datetime, timezone
import os
import pkg_resources

from setuptools import setup
from setuptools import find_packages

VERSION = '0.1.0.dev' + datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M')

SHARED_OBJECT_MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), 'shared_object_list.txt'
)

# Grab the shared objects from the manifest created by the Bazel //tools:shared_object_list target
with open(SHARED_OBJECT_MANIFEST_PATH) as shared_object_manfiest:
  shared_objects = shared_object_manfiest.read().strip().split(' ')

requirements = []

# Get the package dependencies from requirements.in
with open('requirements.in') as requirements_file:
  required_packages = [
      str(requirement)
      for requirement in pkg_resources.parse_requirements(requirements_file)
  ]

setup(
    name='gematria',
    version=VERSION,
    description='Tooling for cost modelling',
    author='Gematria Authors',
    install_requires=requirements,
    packages=find_packages() + [''],
    package_data={'': shared_objects},
    python_requires='>=3.10',
)
