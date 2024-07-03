# Copyright 2024 Google Inc.
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

import lit.formats

config.name = 'convert_bhive_to_llvm_exegesis_input_tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.test']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.obj_root, 'test')

config.substitutions.append(
    ('FileCheck', os.path.join(config.llvm_tools_root, 'FileCheck'))
)
config.substitutions.append(
    ('split-file', os.path.join(config.llvm_tools_root, 'split-file'))
)
config.substitutions.append(
    ('%not', os.path.join(config.llvm_tools_root, 'not'))
)

config.substitutions.append(
    (
        '%convert_bhive_to_llvm_exegesis_input',
        os.path.join(config.tools_root, 'convert_bhive_to_llvm_exegesis_input'),
    )
)
