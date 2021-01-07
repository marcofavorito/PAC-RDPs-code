"""Utility functions."""
import builtins
import importlib.util
import logging
import os
from typing import Any

_default_logger = logging.getLogger(__name__)


def _get_module(spec):
    """Try to execute a module. Return None if the attempt fail."""
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # pylint: disable=broad-except
        return None


def locate(path: str) -> Any:
    """Locate an object by name or dotted save_path, importing as necessary."""
    parts = [part for part in path.split(".") if part]
    module, n = None, 0
    while n < len(parts):
        file_location = os.path.join(*parts[: n + 1])
        spec_name = ".".join(parts[: n + 1])
        module_location = os.path.join(file_location, "__init__.py")
        spec = importlib.util.spec_from_file_location(spec_name, module_location)
        _default_logger.debug("Trying to import {}".format(module_location))
        nextmodule = _get_module(spec)
        if nextmodule is None:
            module_location = file_location + ".py"
            spec = importlib.util.spec_from_file_location(spec_name, module_location)
            _default_logger.debug("Trying to import {}".format(module_location))
            nextmodule = _get_module(spec)

        if nextmodule:
            module, n = nextmodule, n + 1
        else:  # pragma: nocover
            break
    if module:
        object_ = module
    else:
        object_ = builtins
    for part in parts[n:]:
        try:
            object_ = getattr(object_, part)
        except AttributeError:
            return None
    return object_
