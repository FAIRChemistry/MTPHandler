import os
import re
from datetime import datetime
from copy import deepcopy
import numpy as np
from collections import defaultdict

import pandas as pd


def read_magellan(
    cls: "Plate",
    path: str,
    ph: float,
    wavelength: float,
):
    created = datetime.fromtimestamp(os.path.getctime(path)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df = pd.read_excel(path, header=None)

    # Define the format of the input datetime string
    date_format = "%A, %B %d, %Y: %H:%M:%S"
    WELL_ID_PATTERN = r"[A-H][0-9]{1,2}"

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
            temp_value, temp_unit = temperature_str.strip().split("°")
            temperatures.append(float(temp_value))
            time, time_unit = time_str[1:-1].split(" ")

            times.append(time)
            time_unit = time_unit.replace("sec", "s")
            dates.append(datetime.strptime(date_str.strip(), date_format))

    df = df.dropna(how="all")

    for row in df.iterrows():
        if not re.findall(WELL_ID_PATTERN, str(row[1].values[0])):
            continue
        for element in row[1].values:
            if isinstance(element, str):
                key = element
            else:
                data[key].append(element)

    n_rows = 8
    n_columns = 12

    plate = cls(
        date_measured=created,
        n_rows=n_rows,
        n_columns=n_columns,
        measured_wavelengths=[wavelength],
        temperature_unit=temp_unit,
        temperatures=temperatures,
        time_unit=time_unit,
        times=times,
    )

    for well_id, abso_list in data.items():
        x_pos, y_pos = id_to_xy(well_id)
        well = plate.add_to_wells(
            ph=ph,
            id=well_id,
            x_position=x_pos,
            y_position=y_pos,
        )
        well.add_to_measurements(
            wavelength=wavelength,
            wavelength_unit="nm",
            absorptions=abso_list,
            blank_states=[],
        )

    return plate


def id_to_xy(well_id: str):
    return int(well_id[1:]) - 1, ord(well_id[0].upper()) - 65
