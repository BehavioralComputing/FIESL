__all__ = ["FIESL"]


def __getattr__(name: str):
    if name == "FIESL":
        from fiesl.model import FIESL

        return FIESL
    raise AttributeError(name)
