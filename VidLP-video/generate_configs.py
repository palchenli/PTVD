from tvd.config import ex
import json
import os


@ex.automain
def main(_config):
    save_path = os.path.join("configs/", _config["exp_name"] + ".json")
    json.dump(_config, open(save_path, "w"))
    print(save_path)
