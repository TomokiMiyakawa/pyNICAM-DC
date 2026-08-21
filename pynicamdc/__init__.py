
from . import _version
__version__ = _version.get_versions()['version']

__all__ = ["pyNICAM"]


def __getattr__(name):
    # pyNICAM is resolved on first use, not at import: `import pynicamdc` stays free
    # of toml and of api.py, and the prep tools that import this package keep their
    # current import cost. api.py itself imports no model module (see its docstring).
    if name == "pyNICAM":
        from .api import pyNICAM
        return pyNICAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
