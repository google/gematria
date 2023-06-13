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

from absl.testing import absltest

from gematria.datasets.python import bhive_importer
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from gematria.proto import basic_block_pb2
from gematria.proto import canonicalized_instruction_pb2
from gematria.proto import throughput_pb2

_CanonicalizedOperandProto = (
    canonicalized_instruction_pb2.CanonicalizedOperandProto)
_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto)

# A basic block that can be obtained by disassembling the basic block
# "4829d38b44246c8b54246848c1fb034829d04839c3" and using base_address=600.
_EXPECTED_BASIC_BLOCK_PROTO = basic_block_pb2.BasicBlockProto(
    machine_instructions=(
        basic_block_pb2.MachineInstructionProto(
            assembly="\tsubq\t%rdx, %rbx",
            machine_code=b"H)\323",
            address=600,
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tmovl\t108(%rsp), %eax",
            machine_code=b"\213D$l",
            address=603,
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tmovl\t104(%rsp), %edx",
            machine_code=b"\213T$h",
            address=607,
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tsarq\t$3, %rbx",
            machine_code=b"H\301\373\003",
            address=611,
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tsubq\t%rdx, %rax",
            machine_code=b"H)\320",
            address=615,
        ),
        basic_block_pb2.MachineInstructionProto(
            assembly="\tcmpq\t%rax, %rbx",
            machine_code=b"H9\303",
            address=618,
        ),
    ),
    canonicalized_instructions=(
        _CanonicalizedInstructionProto(
            mnemonic="SUB",
            llvm_mnemonic="SUB64rr",
            output_operands=(_CanonicalizedOperandProto(register_name="RBX"),),
            input_operands=(
                _CanonicalizedOperandProto(register_name="RBX"),
                _CanonicalizedOperandProto(register_name="RDX"),
            ),
            implicit_output_operands=(_CanonicalizedOperandProto(
                register_name="EFLAGS"),),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="MOV",
            llvm_mnemonic="MOV32rm",
            output_operands=(_CanonicalizedOperandProto(register_name="EAX"),),
            input_operands=(
                _CanonicalizedOperandProto(
                    memory=_CanonicalizedOperandProto.MemoryLocation(
                        alias_group_id=1)),
                _CanonicalizedOperandProto(
                    address=_CanonicalizedOperandProto.AddressTuple(
                        base_register="RSP", displacement=108, scaling=1)),
            ),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="MOV",
            llvm_mnemonic="MOV32rm",
            output_operands=(_CanonicalizedOperandProto(register_name="EDX"),),
            input_operands=(
                _CanonicalizedOperandProto(
                    memory=_CanonicalizedOperandProto.MemoryLocation(
                        alias_group_id=1)),
                _CanonicalizedOperandProto(
                    address=_CanonicalizedOperandProto.AddressTuple(
                        base_register="RSP", displacement=104, scaling=1)),
            ),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="SAR",
            llvm_mnemonic="SAR64ri",
            output_operands=(_CanonicalizedOperandProto(register_name="RBX"),),
            input_operands=(
                _CanonicalizedOperandProto(register_name="RBX"),
                _CanonicalizedOperandProto(immediate_value=3),
            ),
            implicit_output_operands=(_CanonicalizedOperandProto(
                register_name="EFLAGS"),),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="SUB",
            llvm_mnemonic="SUB64rr",
            output_operands=(_CanonicalizedOperandProto(register_name="RAX"),),
            input_operands=(
                _CanonicalizedOperandProto(register_name="RAX"),
                _CanonicalizedOperandProto(register_name="RDX"),
            ),
            implicit_output_operands=(_CanonicalizedOperandProto(
                register_name="EFLAGS"),),
        ),
        _CanonicalizedInstructionProto(
            mnemonic="CMP",
            llvm_mnemonic="CMP64rr",
            input_operands=(
                _CanonicalizedOperandProto(register_name="RBX"),
                _CanonicalizedOperandProto(register_name="RAX"),
            ),
            implicit_output_operands=(_CanonicalizedOperandProto(
                register_name="EFLAGS"),),
        ),
    ),
)


class BhiveImporterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._x86_llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    self._x86_canonicalizer = canonicalizer.Canonicalizer.x86_64(self._x86_llvm)

  def test_x86_basic_block_proto_from_bytes(self):
    importer = bhive_importer.BHiveImporter(self._x86_canonicalizer)
    block_proto = importer.basic_block_proto_from_bytes(
        b"\x90", base_address=999)
    self.assertEqual(
        block_proto,
        basic_block_pb2.BasicBlockProto(
            machine_instructions=(basic_block_pb2.MachineInstructionProto(
                assembly="\tnop",
                machine_code=b"\x90",
                address=999,
            ),),
            canonicalized_instructions=(_CanonicalizedInstructionProto(
                mnemonic="nop",
                llvm_mnemonic="NOOP",
            ),),
        ),
    )

  def test_x86_basic_block_proto_from_hex(self):
    importer = bhive_importer.BHiveImporter(self._x86_canonicalizer)
    # Basic block:
    #   sub rbx, rdx
    #   mov eax, dword ptr [rsp + 108]
    #   mov edx, dword ptr [rsp + 104]
    #   sar rbx, 3
    #   sub rax, rdx
    #   cmp rbx, rax
    block_proto = importer.basic_block_proto_from_hex(
        "4829d38b44246c8b54246848c1fb034829d04839c3",
        base_address=600,
    )
    self.assertEqual(block_proto, _EXPECTED_BASIC_BLOCK_PROTO)

  def test_x86_parse_csv_line(self):
    source_name = "test: made-up"
    importer = bhive_importer.BHiveImporter(self._x86_canonicalizer)
    # Same basic block as in test_x86_basic_block_proto_from_bytes().
    block_proto = importer.basic_block_with_throughput_proto_from_csv_line(
        source_name=source_name,
        line="4829d38b44246c8b54246848c1fb034829d04839c3,10",
        base_address=600,
        throughput_scaling=2.0,
    )
    self.assertEqual(
        block_proto,
        throughput_pb2.BasicBlockWithThroughputProto(
            basic_block=_EXPECTED_BASIC_BLOCK_PROTO,
            inverse_throughputs=(throughput_pb2.ThroughputWithSourceProto(
                source=source_name,
                inverse_throughput_cycles=[20.0],
            ),),
        ),
    )


if __name__ == "__main__":
  absltest.main()
