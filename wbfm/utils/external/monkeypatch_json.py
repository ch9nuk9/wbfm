""" Module that monkey-patches json module when it's imported so
JSONEncoder.default() automatically checks for a special "to_json()"
method and uses it to encode the object if found.

https://stackoverflow.com/questions/18478287/making-object-json-serializable-with-regular-encoder/18561055
"""
import logging
from json import JSONEncoder


def using_monkeypatch():
    logging.info("Using a monkeypatch to allow my custom objects to be serializable")


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder.default  # Save unmodified default.
JSONEncoder.default = _default  # Replace it.
