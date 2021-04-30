import json
import os
from types import SimpleNamespace


def read_config(config_name: str, args: SimpleNamespace, config_directory: str = 'config') -> SimpleNamespace:
    configs = os.listdir(config_directory)
    configs = [ config for config in configs if os.path.splitext(config)[0] == config_name ]
    if len(configs) != 1:
        raise FileNotFoundError(f'Config {config_name} does not exist in directory {os.path.abspath(config_directory)}')
    config_file = configs[0]
    config_path = os.path.join(config_directory, config_file)
    default_config_path = os.path.join(config_directory, 'default.json')

    current_config = read_single_config(config_path)
    default_config = read_single_config(default_config_path)
    current_config = update_config(current_config, default_config)
    current_config = update_config(current_config, args)
    return current_config
    

def update_config(old_config: SimpleNamespace, new_config: SimpleNamespace) -> SimpleNamespace:
    for k in vars(old_config):
        if hasattr(new_config, k):
            if isinstance(getattr(new_config, k), SimpleNamespace):
                update_config(getattr(old_config, k), getattr(new_config, k))
            else:
                setattr(old_config, k, getattr(new_config, k))
    return old_config


def read_single_config(config_path: str) -> SimpleNamespace:
    config_dict = json.load(open(config_path, 'r'))
    return convert_dict_to_simplenamespace(config_dict)

    
def convert_dict_to_simplenamespace(dic: dict) -> SimpleNamespace:
    sn = SimpleNamespace(**dic)
    for k in vars(sn):
        v = getattr(sn, k)
        if isinstance(v, dict):
            setattr(sn, k, convert_dict_to_simplenamespace(v))
    return sn
