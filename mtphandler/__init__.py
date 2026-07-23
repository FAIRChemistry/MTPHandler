import json  # noqa
import os  # noqa
from importlib.metadata import version

from .mtp_logging import configure_logger
from .plate_manager import PlateManager  # noqa

__version__ = version("mtphandler")

configure_logger()
