import yaml

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config
    
