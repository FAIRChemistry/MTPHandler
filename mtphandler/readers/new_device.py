from __future__ import annotations

import re

import pandas as pd
from mdmodels.units.annotation import UnitDefinitionAnnot

from mtphandler.model import Plate
from mtphandler.readers.utils import id_to_xy


def read_new_device(
    path: str,
    ph: float | None,
    temperature: float,
    temperature_unit: UnitDefinitionAnnot,
) -> Plate:
    pattern = r"\(([A-H][0-9]{2})\)"

    if isinstance(temperature, (int | float)):
        temperature = [temperature]  # type: ignore

    df = pd.read_excel(path)

    wavelength = float(df.iloc[4, 0].split(" ")[1])
    time_str = df.iloc[1, 0]
    timestamp = str(pd.to_datetime(time_str, format="%m/%d/%Y %I:%M:%S %p"))
    name = df.iloc[0, 0]

    # Check for "Reading", "Abs", or "Sample" in the DataFrame
    if "Reading" in df.values:
        plate = handle_timecourse_read(
            df, path, name, timestamp, temperature_unit, wavelength, ph, pattern
        )
    elif any(keyword in df.values for keyword in ["Abs", "Sample"]):
        plate = handle_endpoint_read(
            df, path, name, timestamp, temperature_unit, wavelength, ph, pattern
        )
    else:
        raise ValueError(
            "Could not find 'Reading', 'Abs', or 'Sample' in the DataFrame."
        )

    plate.temperatures = temperature

    return plate


def handle_timecourse_read(
    df: pd.DataFrame,
    path: str,
    name: str,
    timestamp: str,
    temperature_unit: UnitDefinitionAnnot,
    wavelength: float,
    ph: float | None,
    pattern: str,
) -> Plate:
    """Handles the case where 'Reading' is found in the DataFrame."""

    row_number = df[df.iloc[:, 0] == "Reading"].index[0]
    df = pd.read_excel(path, skiprows=row_number + 1).dropna(axis=0, how="any")
    time = df.pop("avg. time [s]").values.tolist()

    plate = Plate(
        name=name,
        date_measured=timestamp,
        temperature_unit=temperature_unit,
    )

    # Iterate over the columns
    for column in df.columns:
        match = re.search(pattern, column)
        if match:
            well_id = match.group(1)
            name_id = column.split(" ")[0]
            x, y = id_to_xy(well_id)
            well = plate.add_to_wells(
                id=column,
                x_pos=x,
                y_pos=y,
                ph=ph,
            )

            well.add_to_measurements(
                wavelength=wavelength,
                absorption=df[column].values.tolist(),
                time=time,
                time_unit="s",
            )

    return plate


def handle_endpoint_read(
    df: pd.DataFrame,
    path: str,
    name: str,
    timestamp: str,
    temperature_unit: UnitDefinitionAnnot,
    wavelength: float,
    ph: float | None,
    pattern: str,
) -> Plate:
    """Handles the case where 'Abs' or 'Sample' is found in the DataFrame."""

    absorption_row = df[df.iloc[:, 0] == "Abs"].index[0]
    sample_row = df[df.iloc[:, 0] == "Sample"].index[0]

    absorption_df = pd.read_excel(
        path,
        skiprows=absorption_row + 1,
        nrows=8,
        usecols=range(13),
        index_col=0,
    )

    sample_df = pd.read_excel(
        path,
        skiprows=sample_row + 1,
        usecols=range(13),
        nrows=8,
        index_col=0,
    )

    plate = Plate(
        name=name,
        date_measured=timestamp,
        temperature_unit=temperature_unit,
    )

    for row_id in range(8):
        for column_id in range(12):
            if pd.isnull(sample_df.iloc[row_id, column_id]) or pd.isnull(
                absorption_df.iloc[row_id, column_id]
            ):
                continue
            well = plate.add_to_wells(
                id=sample_df.iloc[row_id, column_id],
                x_pos=column_id,
                y_pos=row_id,
                ph=ph,
            )

            well.add_to_measurements(
                wavelength=wavelength,
                absorption=[absorption_df.iloc[row_id, column_id]],
                time=[0],
                time_unit="s",
            )

    return plate


if __name__ == "__main__":
    path = "docs/examples/data/new_device_calib.xlsx"

    plate = read_new_device(
        path=path,
        ph=7.4,
        temperature=37.0,
        temperature_unit="celsius",
    )
