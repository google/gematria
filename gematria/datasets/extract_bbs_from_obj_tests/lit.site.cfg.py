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
import os

config.obj_root = os.path.join(
    os.getcwd(), 'gematria/datasets/extract_bbs_from_obj_tests'
)
config.tools_root = os.path.join(os.getcwd(), 'gematria/datasets')
config.llvm_tools_root = os.path.join(os.getcwd(), 'external/llvm-project/llvm')

lit_config.load_config(config, os.path.join(config.obj_root, 'lit.cfg.py'))
