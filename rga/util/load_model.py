import yaml


def load_hparams(params_path):
    with open(params_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_model(params_path, model_path, model_class):
    model = model_class.load_from_checkpoint(
        checkpoint_path=model_path, hparams_file=params_path
    )
    return model
