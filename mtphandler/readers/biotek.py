from __future__ import annotations

import re
from datetime import timedelta

import numpy as np
import pandas as pd

from mtphandler.model import Plate
from mtphandler.readers.utils import id_to_xy
from mtphandler.tools import get_well

PATTERN_WAVELENGTH = r"Wavelengths:\s+(\d{1,4})([\s,;]+\d{1,4})*"


def extract_integers(s: str) -> list[int]:
    # Find all sequences of digits in the string
    matches = re.findall(r"\d+", s)
    # Convert the matched strings to integers
    return [int(match) for match in matches]


def parse_measurement_interval(s: str) -> float:
    # Flexible regex pattern to match time formats
    time_pattern = r"(\d{1,4}):(\d{1,2}):(\d{1,2})"

    interval_match = re.search(r"Interval\s+" + time_pattern, s, re.IGNORECASE)

    if not interval_match:
        raise ValueError("Measurement interval not found.")

    interval = timedelta(
        hours=int(interval_match.group(1)),
        minutes=int(interval_match.group(2)),
        seconds=int(interval_match.group(3)),
    )

    # interval in minutes
    return interval.total_seconds() / 60


def read_biotek(
    path: str,
    ph: float | None,
) -> Plate:
    df = pd.read_excel(path)

    date = get_row_by_value(df, "Date")[-1]
    time = get_row_by_value(df, "Time")[-1]
    timestamp = str(date)
    # timestamp = str(datetime.combine(date, time).isoformat())

    row_index_int_map = df.iloc[:, 0].apply(lambda cell: isinstance(cell, int))
    data_block_starts = [
        index for index, value in enumerate(row_index_int_map) if value
    ]

    wavelengths_cell = str(df.iloc[19, 1])
    if "Absorbance Endpoint" in wavelengths_cell:
        wavelengths_cell = str(df.iloc[20, 1])

    wavelengths = extract_integers(wavelengths_cell)

    measurement_interval = None
    for index, row_value in df.iloc[:, 0].items():
        if isinstance(row_value, str) and "start kinetic" in row_value.lower(): # this is the start of a kinetic read block
            interval_candidate = str(df.iloc[index, 1])
            try:
                measurement_interval = parse_measurement_interval(interval_candidate)
                break # Found a valid interval, stop searching. Note that there could be more kinetic intervals later.
            except ValueError:
                continue # Keep searching if this one fails
    
    if measurement_interval is None:
        # Fallback to original logic if 'Start Kinetic' not found or parsed
        measurement_int_cell = str(df.iloc[15, 1])
        if "add final component" in measurement_int_cell.lower():
            measurement_int_cell = str(df.iloc[16, 1])
        measurement_interval = parse_measurement_interval(measurement_int_cell)

    plate = Plate(
        date_measured=timestamp,
        time_unit="s",
        temperature_unit="C",
    )

    for row_index, (block_start, wavelength) in enumerate(
        zip(data_block_starts, wavelengths)
    ):
        try:
            block = df.iloc[block_start + 2 : data_block_starts[row_index + 1], :]
        except IndexError:
            block = df.iloc[block_start + 2 :, :]

        block = block.drop("Unnamed: 0", axis=1).reset_index(drop=True)

        all_nan_rows = block.isna().all(axis=1)
        first_all_nan_index = all_nan_rows.idxmax() if all_nan_rows.any() else None

        # drop rows if any of the values are NaN
        block.iloc[:first_all_nan_index, :]
        block = block.dropna(how="any", axis=0)
        column_names = block.iloc[0, :].tolist()
        block.columns = column_names
        block = block[1:].reset_index(drop=True)

        # Temperature
        temperature = block.pop(column_names[1]).values

        for column_name in column_names[2:]:
            x, y = id_to_xy(column_name)

            try:
                well = get_well(plate, column_name)
            except ValueError:
                well = plate.add_to_wells(id=column_name, x_pos=x, y_pos=y, ph=ph)

            data = block[column_name].values.tolist()
            time = np.arange(
                0, len(data) * measurement_interval, measurement_interval
            ).tolist()

            well.add_to_measurements(
                wavelength=wavelength,
                absorption=data,
                time=time,
                time_unit="s",
            )

            plate.temperatures = temperature.tolist()
            plate.times = time

    # assert that all plate -> well -> measurement -> absorption have the same length
    for well in plate.wells:
        for measurement in well.measurements:
            assert len(measurement.absorption) == len(measurement.time), (
                f"Absorption and time data for well {well.id} and wavelength "
                f"{measurement.wavelength} do not have the same length."
            )

    return plate


def get_row_by_value(df: pd.DataFrame, value: str) -> list:
    row_df = df[df.iloc[:, 0].values == value]
    row_df = row_df.reset_index(drop=True)
    row_df = row_df.dropna(axis=1, how="all")
    return row_df.loc[0].values.tolist()


if __name__ == "__main__":
    from devtools import pprint

    path = "docs/examples/data/20240719_biotek_FDH.V9_0.02enzyme_kinetics.xlsx"

    p = read_biotek(path, ph=7.4)
    pprint(p.wells[0])
