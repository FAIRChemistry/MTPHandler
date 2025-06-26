from __future__ import annotations

import re
from datetime import datetime
from typing import Union

import pandas as pd
from typing_extensions import Optional

from mtphandler.model import Plate
from mtphandler.readers.utils import id_to_xy

# Constants
LUMINESCENCE_WAVELENGTH = 0
TEKAN_DEVICE_IDENTIFIER = "Device: Spark"


def _extract_metadata_value(
    meta_section: pd.DataFrame, field_prefix: str
) -> str | None:
    """Extract a metadata value from the meta section by field prefix."""
    for _, row in meta_section.iterrows():
        first_col = str(row.iloc[0]).strip()
        if first_col.startswith(field_prefix):
            for col_idx in range(1, len(row)):
                if pd.notna(row.iloc[col_idx]) and str(row.iloc[col_idx]).strip():
                    return str(row.iloc[col_idx]).strip()
    return None


def _extract_temperature(meta_section: pd.DataFrame) -> float:
    """Extract temperature from metadata. Raises error if not found."""
    temp_str = _extract_metadata_value(meta_section, "Temperature")
    if temp_str:
        temp_match = re.search(r"(\d+(?:\.\d+)?)", temp_str)
        if temp_match:
            return float(temp_match.group(1))

    raise ValueError(
        "Temperature not found in metadata. Temperature is required for endpoint measurements."
    )


def _parse_timestamp(meta_section: pd.DataFrame) -> datetime:
    """Parse date and time from metadata, with fallback to current time."""
    date_measured = _extract_metadata_value(meta_section, "Date:")
    time_measured = _extract_metadata_value(meta_section, "Time:")

    if date_measured and time_measured:
        try:
            return datetime.strptime(
                f"{date_measured} {time_measured}", "%d.%m.%Y %H:%M:%S"
            )
        except ValueError:
            pass

    return datetime.now()


def _detect_measurement_type(df: pd.DataFrame) -> Union[int, str]:
    """Detect measurement type and return appropriate wavelength. Raises error if not found."""
    for _, row in df.iterrows():
        row_content = str(row.iloc[1])
        if "Luminescence" in row_content:
            return LUMINESCENCE_WAVELENGTH
        elif "Absorbance" in row_content:
            # Try to extract wavelength from nearby content
            wavelength_match = re.search(r"(\d+)\s*nm", str(row))
            if wavelength_match:
                return int(wavelength_match.group(1))
            else:
                raise ValueError(
                    "Absorbance measurement detected but wavelength not found. Wavelength is required."
                )

    raise ValueError(
        "Measurement type (Luminescence/Absorbance) not found in metadata. Measurement type is required."
    )


def _extract_plate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean plate data from the raw DataFrame."""
    hdr = df.index[df.iloc[:, 0].eq("<>")][0]
    end = df.index[df.iloc[:, 0].eq("End Time")][0]

    plate_data = df.iloc[hdr : end - 2].copy()
    plate_data.columns = plate_data.iloc[0]
    plate_data = plate_data.iloc[1:]

    return (
        plate_data.rename(columns={"<>": "row"})
        .set_index("row")
        .dropna(axis=1, how="all")
        .apply(pd.to_numeric, errors="coerce")
    )


def _parse_endpoint_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, Union[int, str], datetime, float]:
    """Parse endpoint (single timepoint) Tekan Spark data."""
    plate_data = _extract_plate_data(df)

    # Extract metadata
    hdr = df.index[df.iloc[:, 0].eq("<>")][0]
    meta_section = df.iloc[:hdr, :]

    timestamp = _parse_timestamp(meta_section)
    temperature = _extract_temperature(meta_section)
    wavelength = _detect_measurement_type(df)

    return plate_data, wavelength, timestamp, temperature


def _parse_kinetic_metadata(
    df: pd.DataFrame, cycle_no_row_index: int
) -> tuple[datetime, Union[int, str]]:
    """Parse metadata from kinetic data."""
    meta_df = (
        df.iloc[:cycle_no_row_index, :]
        .dropna(how="all")
        .dropna(axis=1, how="all")
        .set_index(df.columns[0])
    )

    time_measured_str = meta_df.loc["Start Time"].dropna(axis=1, how="all").values[0][0]
    time_measured = datetime.strptime(time_measured_str, "%Y-%m-%d %H:%M:%S")

    # Validate wavelength data
    try:
        wavelength = meta_df.loc["Measurement wavelength"].dropna().iloc[0]
        if pd.isna(wavelength) or wavelength == "":
            raise ValueError(
                "Wavelength data is empty or invalid in kinetic measurements. Wavelength is required."
            )
    except KeyError:
        raise ValueError(
            "Measurement wavelength not found in kinetic metadata. Wavelength is required."
        )

    return time_measured, wavelength


def _parse_kinetic_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, Union[int, str], datetime]:
    """Parse kinetic (time series) Tekan Spark data."""
    cycle_no_row_index = df[df.iloc[:, 0].str.contains("Cycle Nr.", na=False)].index[0]
    time_measured, wavelength = _parse_kinetic_metadata(df, cycle_no_row_index)

    # Parse data section
    data_df = df.iloc[cycle_no_row_index:, :].reset_index(drop=True)
    data_df = data_df.set_index(data_df.columns[0])
    data_df.columns = data_df.iloc[0, :].tolist()
    data_df = data_df[1:].reset_index(drop=True).dropna(axis=1, how="all")

    # Trim to valid data
    first_nan_index = data_df.isna().any(axis=1).idxmax()
    data_df = data_df.iloc[:first_nan_index, :]

    time_series = data_df.pop("Time [s]") / 60
    temp_series = data_df.pop("Temp. [°C]")

    return data_df, time_series, temp_series, wavelength, time_measured


def _create_endpoint_plate(
    plate_data: pd.DataFrame,
    wavelength: Union[int, str],
    timestamp: datetime,
    temperature: float,
    ph: Optional[float],
) -> Plate:
    """Create a Plate object from endpoint data."""
    plate = Plate(
        date_measured=str(timestamp),
        temperatures=[temperature],
        temperature_unit="C",
        time_unit="s",
        times=[0.0],
    )

    for row_letter, row_data in plate_data.iterrows():
        if pd.isna(row_letter):
            continue

        for col_num in plate_data.columns:
            if pd.isna(col_num):
                continue

            measurement_value = row_data[col_num]
            if pd.isna(measurement_value):
                continue

            well_id = f"{row_letter}{int(col_num)}"
            x, y = id_to_xy(well_id)

            well = plate.add_to_wells(id=well_id, x_pos=x, y_pos=y, ph=ph)
            well.add_to_measurements(
                wavelength=wavelength,
                wavelength_unit="nm" if wavelength != LUMINESCENCE_WAVELENGTH else "",
                absorption=[float(measurement_value)],
                time=[0.0],
                time_unit="s",
            )

    return plate


def _create_kinetic_plate(
    data_df: pd.DataFrame,
    time_series: pd.Series,
    temp_series: pd.Series,
    wavelength: Union[int, str],
    timestamp: datetime,
    ph: Optional[float],
) -> Plate:
    """Create a Plate object from kinetic data."""
    # Validate temperature data
    if temp_series.empty or temp_series.isna().all():
        raise ValueError(
            "Temperature data not found or invalid in kinetic measurements. Temperature is required."
        )

    plate = Plate(
        date_measured=str(timestamp),
        temperatures=temp_series.values.tolist(),
        temperature_unit="C",
        time_unit="s",
        times=time_series.values.tolist(),
    )

    for column in data_df.columns:
        x, y = id_to_xy(column)
        well = plate.add_to_wells(id=column, x_pos=x, y_pos=y, ph=ph)
        well.add_to_measurements(
            wavelength=wavelength,
            wavelength_unit="nm",
            absorption=data_df[column].values.tolist(),
            time=time_series.values.tolist(),
            time_unit="s",
        )

    return plate


def read_tekan_spark(path: str, ph: Optional[float]) -> Plate:
    """
    Read Tekan Spark plate reader data (both kinetic and endpoint).

    Args:
        path: Path to the Excel file
        ph: pH value for the measurements

    Returns:
        Plate object containing the parsed data

    Raises:
        ValueError: If the file is not a valid Tekan Spark file
    """
    df = pd.read_excel(path)

    if not df.iloc[1, 0] == TEKAN_DEVICE_IDENTIFIER:
        raise ValueError("The file does not seem to be a Tekan Spark file.")

    # Detect data type: kinetic vs endpoint
    has_cycle_nr = df.iloc[:, 0].str.contains("Cycle Nr.", na=False).any()

    if has_cycle_nr:
        data_df, time_series, temp_series, wavelength, timestamp = _parse_kinetic_data(
            df
        )
        return _create_kinetic_plate(
            data_df, time_series, temp_series, wavelength, timestamp, ph
        )
    else:
        plate_data, wavelength, timestamp, temperature = _parse_endpoint_data(df)
        return _create_endpoint_plate(
            plate_data, wavelength, timestamp, temperature, ph
        )


if __name__ == "__main__":
    from rich import print

    # Test endpoint data
    endpoint_path = "docs/examples/data/tekan_spark_endpoint.xlsx"
    endpoint_plate = read_tekan_spark(endpoint_path, ph=7.4)
    print("=== ENDPOINT PLATE ===")
    print(
        f"Wells: {len(endpoint_plate.wells)}, Temperature: {endpoint_plate.temperatures[0]}°C"
    )

    # Test kinetic data
    kinetic_path = "docs/examples/data/tekan_spark.xlsx"
    kinetic_plate = read_tekan_spark(kinetic_path, ph=7.4)
    print("\n=== KINETIC PLATE ===")
    print(f"Wells: {len(kinetic_plate.wells)}, Timepoints: {len(kinetic_plate.times)}")
