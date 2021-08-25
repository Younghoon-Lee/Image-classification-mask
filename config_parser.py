import argparse
import json
from dataclasses import dataclass

@dataclass
class ConfigParser(argparse.ArgumentParser):
    # basic settings
    data_dir: str = ''
    eval_dir_name: str = ''
    train_dir_name: str = ''

    # hyperparameters
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3

    def __init__(self, description):
        super().__init__(description=description)
        self.add_argument('--config', metavar='-C', help='a')
        self._set_config()
    
    def _set_config(self):
        _args_dict = vars(self.parse_args())
        with open(_args_dict['config'], 'r') as config_json:
            contents = config_json.read()
            configs = json.loads(contents)

            for arg in self.__annotations__.keys():
                exec(f'self.{arg} = configs["config"][arg]')
