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

from absl.testing import absltest
from gematria.datasets.python import annotating_importer
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from gematria.proto import basic_block_pb2
from gematria.proto import canonicalized_instruction_pb2
from gematria.proto import throughput_pb2
from rules_python.python.runfiles import runfiles

_CanonicalizedOperandProto = (
    canonicalized_instruction_pb2.CanonicalizedOperandProto
)
_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto
)


_EXPECTED_BASIC_BLOCK_PROTO = basic_block_pb2.BasicBlockProto(
    machine_instructions=(
        basic_block_pb2.MachineInstructionProto(
            assembly="\tmovl\t%ecx, %edx",
            address=18446744073709547787,
            machine_code=b"\211\312",
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\timull\t%edx, %edx",
            address=18446744073709547789,
            machine_code=b"\017\257\322",
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\taddl\t%edx, %eax",
            address=18446744073709547792,
            machine_code=b"\001\320",
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tdecl\t%ecx",
            address=18446744073709547794,
            machine_code=b"\377\311",
        ),
    ),
    canonicalized_instructions=(
        _CanonicalizedInstructionProto(
            mnemonic="MOV",
            llvm_mnemonic="MOV32rr",
            output_operands=(_CanonicalizedOperandProto(register_name="EDX"),),
            input_operands=(_CanonicalizedOperandProto(register_name="ECX"),),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="IMUL",
            llvm_mnemonic="IMUL32rr",
            output_operands=(_CanonicalizedOperandProto(register_name="EDX"),),
            input_operands=(
                _CanonicalizedOperandProto(register_name="EDX"),
                _CanonicalizedOperandProto(register_name="EDX"),
            ),
            implicit_output_operands=(
                _CanonicalizedOperandProto(register_name="EFLAGS"),
            ),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="ADD",
            llvm_mnemonic="ADD32rr",
            output_operands=(_CanonicalizedOperandProto(register_name="EAX"),),
            input_operands=(
                _CanonicalizedOperandProto(register_name="EAX"),
                _CanonicalizedOperandProto(register_name="EDX"),
            ),
            implicit_output_operands=(
                _CanonicalizedOperandProto(register_name="EFLAGS"),
            ),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="DEC",
            llvm_mnemonic="DEC32r",
            output_operands=(_CanonicalizedOperandProto(register_name="ECX"),),
            input_operands=(_CanonicalizedOperandProto(register_name="ECX"),),
            implicit_output_operands=(
                _CanonicalizedOperandProto(register_name="EFLAGS"),
            ),
        ),
    ),
)


class AnnotatingImporterTest(absltest.TestCase):

  _ELF_OBJECT_FILEPATH = (
      r"com_google_gematria/gematria/testing/testdata/simple_x86_elf_object"
  )
  _PERF_DATA_FILEPATH = (
      r"com_google_gematria/gematria/testing/testdata/"
      r"simple_x86_elf_object.perf.data"
  )
  _SOURCE_NAME = "test: skl"

  def setUp(self):
    super().setUp()

    self._x86_llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    self._x86_canonicalizer = canonicalizer.Canonicalizer.x86_64(self._x86_llvm)

    self._runfiles = runfiles.Create(
        {"RUNFILES_DIR": os.environ.get("PYTHON_RUNFILES")}
    )
    assert self._runfiles is not None

  def test_x86_basic_block_proto_from_binary_and_profile(self):
    source_name = "test: skl"
    importer = annotating_importer.AnnotatingImporter(self._x86_canonicalizer)
    block_protos = importer.get_annotated_basic_block_protos(
        elf_file_name=self._runfiles.Rlocation(self._ELF_OBJECT_FILEPATH),
        perf_data_file_name=self._runfiles.Rlocation(self._PERF_DATA_FILEPATH),
        source_name=self._SOURCE_NAME,
    )
    self.assertEqual(len(block_protos), 1)
    self.assertEqual(
        block_protos[0],
        throughput_pb2.BasicBlockWithThroughputProto(
            basic_block=_EXPECTED_BASIC_BLOCK_PROTO,
            inverse_throughputs=(
                throughput_pb2.ThroughputWithSourceProto(
                    source=source_name,
                    inverse_throughput_cycles=[1.532258064516129],
                ),
            ),
        ),
    )


if __name__ == "__main__":
  absltest.main()
