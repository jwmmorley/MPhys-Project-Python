__all__ = ['processing']

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    from setuptools_scm import get_version
    __version__ = get_version()

from flask import Flask
app = Flask(__name__)
app.config.from_pyfile('config.cfg')

from . import views
