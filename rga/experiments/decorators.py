from argparse import ArgumentParser


def add_model_specific_args(cls, parser: ArgumentParser):
    for base in cls.__bases__:
        parser = base.add_model_specific_args(parser)
    parser = cls.graphloader_class.add_model_specific_args(parser)
    return parser


def add_graphloader_args(cls):
    cls.add_model_specific_args = classmethod(add_model_specific_args)
    return cls
