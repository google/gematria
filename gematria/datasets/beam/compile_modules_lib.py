import apache_beam as beam
from rules_python.python.runfiles import runfiles

from collections.abc import Iterable
from collections.abc import Callable

import subprocess

class OptimizeModules(beam.DoFn):
  def __init__(self, optimization_pass_lists: list[str]):
    self.optimization_pass_lists = optimization_pass_lists
  
  def optimize_module(input_module: bytes, optimization_pass_list: list[str]) -> bytes:
    command_vector = ['opt', f'passes={optimization_pass_list}']
    with subprocess.Popen(command_vector, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as opt_process:
      (output_bc, stderr) = opt_process.communicate(input=input_module)
      del stderr # We do not need any stderr output.
      opt_process.wait()
      if opt_process.returncode != 0:
        raise ValueError("Expected opt to return 0")
      return output_bc
  
  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_pass_list in self.optimization_pass_lists:
      yield self.optimize_module(input_module, optimization_pass_list)

class LowerModulesAsm(beam.DoFn):
  def __init__(self, optimization_levels: list[str]):
    self.optimization_levels = optimization_levels

class GetBBsFromModule(beam.DoFn):
  def process(self, input_object_file: bytes) -> Iterable[str]:
    return ["aa"]

def get_bbs(input_file_pattern: str,
            output_file: str) -> Callable[[beam.Pipeline], None]:
  
  def pipeline(root: beam.Pipeline) -> None:
    print('test')
  
  return pipeline
