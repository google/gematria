import lit.formats

config.name = 'gematria'
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

config.substitutions.append((
    '%convert_bhive_to_llvm_exegesis_input',
    os.path.join(config.tools_root, 'convert_bhive_to_llvm_exegesis_input'),
))
