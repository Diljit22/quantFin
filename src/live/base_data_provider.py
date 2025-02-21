import os
import yaml
import datetime

class BaseDataProvider:
    """
    Base class for data providers.
    
    This class loads configuration files (secrets and provider settings) and
    provides a helper method for saving artifacts (e.g. API call responses) to disk.
    """
    
    def __init__(self, **kwargs):
        # Load secrets from a YAML file; default file: config/secrets.yaml
        self.secrets = self.load_config("config/secrets.yaml")
        # Load provider settings from a YAML file; default file: config/provider_settings.yaml
        self.provider_settings = self.load_config("config/provider_settings.yaml")
        
        # Allow overriding via kwargs.
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def load_config(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                try:
                    return yaml.safe_load(file)
                except Exception as e:
                    raise ValueError(f"Error reading config file {filepath}: {e}")
        else:
            # If the file doesn't exist, return an empty dict.
            return {}
    
    def save_artifact(self, folder: str, filename: str, data: dict) -> None:
        """
        Save artifact data (a dictionary) to a file in CSV format.
        
        The file will be appended with a timestamp line.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        timestamp = datetime.datetime.now().isoformat()
        # Create a CSV line: timestamp, key1=value1, key2=value2, ...
        line = timestamp + "," + ",".join(f"{k}={v}" for k, v in data.items()) + "\n"
        with open(filepath, "a") as f:
            f.write(line)
