import sys
import yaml

config_file = open(sys.path[0] + "/config.yaml")
CONFIG = yaml.load(config_file)
config_file.close()
