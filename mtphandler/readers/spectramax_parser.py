from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
from mdmodels.units.annotation import UnitDefinitionAnnot

from mtphandler.model import Plate, Well


def read_spectramax(
    path: str,
    time_unit: UnitDefinitionAnnot,
    ph: float | None = None,
):
    df = pd.read_csv(
        path,
        sep="delimiter",
        encoding="utf-16",
        engine="python",
        skiprows=15,
    )

    df = df.map(lambda x: x.split("\t"))

    # Get date of measurement
    last_saved = re.findall(
        r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} [APMapm]{2}", df.iloc[-1, 0][0]
    )[0]
    created = datetime.strptime(last_saved, "%Y/%m/%d %I:%M:%S %p").isoformat()

    wavelengths = df.iloc[0, 0][-6]
    wavelengths = [
        float(wavelength) for wavelength in wavelengths.split(" ") if wavelength != ""
    ]

    # data
    datas = df.iloc[2:-9]["~End"].tolist()
    time_pattern = r"\d{2}:\d{2}:\d{2}"
    times = []
    temperatures = []
    block_start_ids = []
    for data in datas:
        if re.match(time_pattern, data[0]):
            times.append(to_time(data[0], time_unit))
            temperatures.append(float(data[1]))
            block_start_ids.append(datas.index(data))

    time_blocks = []
    for index, start_id in enumerate(block_start_ids):
        try:
            time_block = datas[start_id : block_start_ids[index + 1]]
        except IndexError:
            time_block = datas[start_id:]

        time_block[0] = time_block[0][2:]  # remove time and temperature from block
        for row_id, row in enumerate(time_block):
            if "" not in row:
                continue

            wavelength_entry = []
            wavelength_entries = []
            for item in row:
                if item == "":
                    if wavelength_entry:
                        wavelength_entries.append(wavelength_entry)
                        wavelength_entry = []  # reset

                else:
                    wavelength_entry.append(item)

            if item != "":
                wavelength_entries.append(wavelength_entry)

            time_block[row_id] = wavelength_entries

        time_blocks.append(time_block)

    # Swap dimensions: rows, columns, wavelengths, timecourse
    data = np.array(time_blocks).astype(float)
    data = data.swapaxes(0, 3)
    data = data.swapaxes(0, 1)

    # create wells
    wells = []
    for row_id, row in enumerate(data):
        for column_id, column in enumerate(row):
            well = Well(
                id=_coordinates_to_id(column_id, row_id),
                ph=ph,
                x_pos=column_id,
                y_pos=row_id,
            )
            for wavelength_id, wavelength in enumerate(column):
                well.add_to_measurements(
                    wavelength=wavelengths[wavelength_id],
                    wavelength_unit="nm",
                    absorption=wavelength.tolist(),
                    time=times,
                    time_unit=time_unit,
                )
            wells.append(well)

    # Create plate
    plate = Plate(
        date_measured=created,
        time_unit=time_unit,
        temperatures=temperatures,
        temperature_unit="C",
        wells=wells,
    )

    return plate


def _coordinates_to_id(x: int, y: int) -> str:
    return f"{chr(y + 65)}{x + 1}"
