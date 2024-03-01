import os

config.obj_root = os.path.join(os.getcwd(), 'gematria/datasets/tests')
config.tools_root = os.path.join(os.getcwd(), 'gematria/datasets')
config.llvm_tools_root = os.path.join(os.getcwd(), 'external/llvm-project/llvm')

lit_config.load_config(config, os.path.join(config.obj_root, 'lit.cfg.py'))
