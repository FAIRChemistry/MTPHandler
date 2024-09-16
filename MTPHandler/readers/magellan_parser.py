from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

from MTPHandler.dataclasses import Plate
from MTPHandler.readers.utils import WELL_ID_PATTERN, id_to_xy
from MTPHandler.units import C, nm, s

LOGGER = logging.getLogger(__name__)


def read_magellan(
    path: str,
    wavelength: float,
    ph: Optional[float] = None,
):
    df = pd.read_excel(path, header=None)

    # Define the format of the input datetime string
    date_format = "%A, %B %d, %Y: %H:%M:%S"

    data = defaultdict(list)
    temperatures = []
    times = []
    dates = []
    for row in df.iterrows():
        timecourser_data = row[1].values[0]
        if not isinstance(timecourser_data, str):
            break
        else:
            date_str, time_str, temperature_str = timecourser_data.split("/")
            temp_value, _ = temperature_str.strip().split("°")
            temperatures.append(float(temp_value))
            time, time_unit = time_str[1:-1].split(" ")

            times.append(time)
            dates.append(datetime.strptime(date_str.strip(), date_format))

    created = dates[0]

    df = df.dropna(how="all")

    for row in df.iterrows():
        first_cell = str(row[1].values[0])
        if not re.findall(WELL_ID_PATTERN, first_cell):
            continue
        for element in row[1].values:
            if isinstance(element, str):
                key = element
            elif math.isnan(element):
                continue
            else:
                data[key].append(element)

    plate = Plate(
        date_measured=str(created),
        temperature_unit=C,
        temperatures=temperatures,
        time_unit=s,
        times=times,
    )

    for well_id, abso_list in data.items():
        x_pos, y_pos = id_to_xy(well_id)
        well = plate.add_to_wells(
            ph=ph if ph else None,
            id=well_id,
            x_pos=x_pos,
            y_pos=y_pos,
        )
        well.add_to_measurements(
            wavelength=wavelength,
            wavelength_unit=nm,
            absorption=abso_list,
            time_unit=s,
            time=times,
        )

    return plate


if __name__ == "__main__":
    path = "tests/data/magellan.xlsx"
    from devtools import pprint

    plate = read_magellan(path, wavelength=600, ph=7)
    pprint(plate)
