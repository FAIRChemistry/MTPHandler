from __future__ import annotations

import pandas as pd

from mtphandler.model import Plate, Well
from mtphandler.readers.utils import id_to_xy


def read_multiskan_sky(
    path: str,
    ph: float | None,
) -> Plate:
    sheetnames = pd.ExcelFile(path).sheet_names

    RAW_DATA = next(sheet for sheet in sheetnames if "Raw data" in sheet), None
    GENERAL_INFO = next(sheet for sheet in sheetnames if "General info" in sheet), None

    if RAW_DATA is None or GENERAL_INFO is None:
        raise ValueError(
            "The provided Excel file does not contain the expected sheets."
        )

    df, timestamp = raw_data_to_df(RAW_DATA[0], path)
    temperature = get_temperature(GENERAL_INFO[0], path)

    wells = df_to_wells(df, ph)

    return Plate(
        wells=wells,
        date_measured=timestamp,
        temperatures=[temperature],
        temperature_unit="C",
    )


def get_temperature(sheetname: str, path: str) -> float:
    df = pd.read_excel(path, sheet_name=sheetname)

    return float(df.iloc[6, 3].split(" ")[0])


def raw_data_to_df(sheetname: str, path: str) -> pd.DataFrame:
    # Read the sheet from the Excel file
    df = pd.read_excel(path, sheet_name=sheetname)

    # Extract the timestamp from the first row
    timestamp = str(df.iloc[0, 0])

    # Identify the row number where 'Well' is located
    well_row = df[df.iloc[:, 0] == "Well"].index[0]

    # Extract the data from the identified row onwards
    data_df = df.iloc[well_row:, :]

    # Set the first row as column names
    data_df.columns = data_df.iloc[0]

    # Drop the row with the column names as it is now set as the header
    data_df = data_df.drop(data_df.index[0])

    # Reset the index and set a MultiIndex with 'Well' and 'Wavelength(s) [nm]'
    data_df.set_index(["Well", "Wavelength(s) [nm]"], inplace=True)

    return data_df, timestamp


def df_to_wells(df: pd.DataFrame, ph: float | None) -> pd.DataFrame:
    wells = []
    existing_well_ids = set()

    for well_id in df.index.get_level_values("Well").unique():
        df_well = df.loc[well_id]
        well_id = well_id.replace(" ", "").strip()

        for wavelength in df_well.index.get_level_values("Wavelength(s) [nm]").unique():
            df_wavelength = df_well.loc[wavelength]
            df_wavelength = df_wavelength.sort_values(by="Measurement time(s)")

            if well_id not in existing_well_ids:
                x, y = id_to_xy(well_id)
                well = Well(
                    id=well_id,
                    x_pos=x,
                    y_pos=y,
                    ph=ph,
                )
                existing_well_ids.add(well_id)

            well.add_to_measurements(
                wavelength=wavelength,
                absorption=df_wavelength["Raw absorbance"].tolist(),
                time=df_wavelength["Measurement time(s)"] / 60,
                time_unit="s",
            )

        wells.append(well)

    return wells


if __name__ == "__main__":
    path = (
        "/Users/max/Documents/GitHub/MTPHandler/docs/examples/data/Multiskan Sky.xlsx"
    )

    print(read_multiskan_sky(path, ph=None))
