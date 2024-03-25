import lit.formats

config.name = 'extract_bbs_from_obj_tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.test']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.obj_root, 'test')

config.substitutions.append(
    ('FileCheck', os.path.join(config.llvm_tools_root, 'FileCheck'))
)
config.substitutions.append(
    ('%yaml2obj', os.path.join(config.llvm_tools_root, 'yaml2obj'))
)

config.substitutions.append((
    '%extract_bbs_from_obj',
    os.path.join(config.tools_root, 'extract_bbs_from_obj')
))
