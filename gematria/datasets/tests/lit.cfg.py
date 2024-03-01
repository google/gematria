import lit.formats

config.name = 'gematria'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.csv']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.obj_root, 'test')
