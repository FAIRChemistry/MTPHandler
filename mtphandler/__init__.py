import json  # noqa
import os  # noqa

from .mtp_logging import configure_logger
from .plate_manager import PlateManager  # noqa


configure_logger()

__all__ = [
    "PlateManager",
]

if __name__ == "__main__":
    # Quick sanity check for Tecan Infinite reader (adjust path if needed)
    from mtphandler.readers import read_tekan_infinity

    try:
        path = "docs/examples/data/tekan_infinity.xlsx"
        plate = read_tekan_infinity(path, ph=7.4)
        print(
            f"Quick-check: loaded plate with {len(plate.wells)} wells and {len(plate.times)} timepoints"
        )
    except Exception as exc:
        print(f"Quick-check failed: {exc}")
