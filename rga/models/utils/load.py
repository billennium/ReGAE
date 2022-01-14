import yaml


def load_hparams(hparams_path):
    with open(hparams_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_model(hparams_path, model_path, model_class):
    model = model_class.load_from_checkpoint(
        checkpoint_path=model_path, hparams_file=hparams_path
    )
    return model
