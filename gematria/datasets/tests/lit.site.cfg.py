import os

config.src_root = r'./'
config.obj_root = r'./'

lit_config.load_config(
        config, os.path.join(config.src_root, "lit.cfg.py"))
