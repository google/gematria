# Copyright 2022 Google Inc.
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
"""Definitions of architecture-independent tokens for canonicalized code."""

# Structural tokens that are neither instructions, instruction prefixes or
# registers. These tokens can't be extracted from the assembly code, and need to
# be maintained by hand.
DELIMITER = '_D_'
IMMEDIATE = '_IMMEDIATE_'
ADDRESS = '_ADDRESS_'
MEMORY = '_MEMORY_'
NO_REGISTER = '_NO_REGISTER_'
DISPLACEMENT = '_DISPLACEMENT_'
UNKNOWN = '_UNKNOWN_'

# The list of structural tokens that may be added to the models.
STRUCTURAL_TOKENS = (
    DELIMITER,
    IMMEDIATE,
    ADDRESS,
    MEMORY,
    NO_REGISTER,
    DISPLACEMENT,
    UNKNOWN,
)
