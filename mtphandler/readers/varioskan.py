from __future__ import annotations

import re

import pandas as pd

from mtphandler.model import Plate, Well
from mtphandler.readers.utils import id_to_xy

_WELL_HEADER = re.compile(r"\(([A-H])(\d{1,2})\)", re.IGNORECASE)
_WAVELENGTH_LINE = re.compile(
    r"Wavelength:\s*(\d+(?:[.,]\d+)?)\s*nm", re.IGNORECASE
)
_ROW_LABEL = re.compile(r"[A-H]")
_TEMPERATURE_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")


def read_varioskan(path: str, ph: float | None = None) -> Plate:
    """Read a SkanIt `*.xlsx` export from a Varioskan ALF.

    Endpoint grids, kinetic time courses, and absorbance spectra are accepted.
    Calculated sheets such as dilution factor and blank subtraction are ignored.
    Temperature is taken from the instrument setpoint or the run log. A measurement-only
    export, such as a spectrum without those sheets, leaves the temperature empty.

    Args:
        path: Path to the Excel export.
        ph: pH written onto each well. The export does not contain pH.

    Returns:
        Plate with raw absorbance, one measurement per well and wavelength.
    """
    workbook = pd.ExcelFile(path)
    _assert_varioskan(workbook, path)
    name, timestamp = _session(workbook, path)
    temperature = _temperature(workbook, path)

    plate = Plate(
        name=name,
        date_measured=timestamp,
        temperatures=[] if temperature is None else [temperature],
        temperature_unit="C",
    )

    absorbance_sheets = [
        sheet for sheet in workbook.sheet_names if sheet.startswith("Absorbance")
    ]
    if not absorbance_sheets:
        raise ValueError(f"No absorbance sheet found in {path}.")

    for sheet in absorbance_sheets:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        layout = _detect_layout(frame)
        if layout == "endpoint":
            _parse_endpoint(frame, plate, ph)
        elif layout == "kinetic":
            _parse_kinetic(frame, plate, ph)
        else:
            _parse_spectrum(frame, plate, ph)

    if not plate.wells:
        raise ValueError(f"No wells found in {path}.")

    return plate


def _assert_varioskan(workbook: pd.ExcelFile, path: str) -> None:
    sheet = _find_sheet(workbook, "Instrument information")
    if sheet is not None:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        instrument_name = _value_beside_label(frame, "Name")
        if instrument_name is None or "varioskan" not in instrument_name.lower():
            raise ValueError(
                f"Expected a Varioskan export, found instrument {instrument_name!r}."
            )
        return

    for sheet_name in workbook.sheet_names:
        if not sheet_name.startswith("Absorbance"):
            continue
        frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if _text(frame.iat[0, 0]) == "Measurement results":
            return

    raise ValueError(f"Expected a Varioskan SkanIt export, got {path}.")


def _session(workbook: pd.ExcelFile, path: str) -> tuple[str, str]:
    sheet = _find_sheet(workbook, "Session information")
    if sheet is not None:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        name = _value_beside_label(frame, "Session name")
        timestamp = _value_beside_label(frame, "Execution time")
        if name and timestamp:
            return name, timestamp

    frame = _first_absorbance_frame(workbook, path)
    name = _text(frame.iat[1, 0])
    timestamp = _text(frame.iat[2, 0])
    if not name or not timestamp:
        raise ValueError(f"Could not find session name and time in {path}.")
    return name, timestamp


def _temperature(workbook: pd.ExcelFile, path: str) -> float | None:
    instrument_sheet = _find_sheet(workbook, "Instrument information")
    run_log_sheet = _find_sheet(workbook, "Run log")
    if instrument_sheet is None and run_log_sheet is None:
        return None

    if instrument_sheet is not None:
        instrument = pd.read_excel(path, sheet_name=instrument_sheet, header=None)
        setpoint = _value_beside_label(instrument, "Set temperature")
        if setpoint is not None:
            return _parse_temperature(setpoint)

    if run_log_sheet is not None:
        run_log = pd.read_excel(path, sheet_name=run_log_sheet, header=None)
        for row_index in range(len(run_log)):
            event = _text(run_log.iat[row_index, 2] if run_log.shape[1] > 2 else None)
            if event != "Temperature":
                continue
            information = run_log.iat[row_index, 3] if run_log.shape[1] > 3 else None
            if _text(information):
                return _parse_temperature(_text(information))

    raise ValueError(f"No temperature found in {path}.")


def _parse_temperature(text: str) -> float:
    match = _TEMPERATURE_NUMBER.search(text)
    if match is None:
        raise ValueError(f"Could not parse temperature from {text!r}.")
    return float(match.group(1).replace(",", "."))


def _parse_endpoint(frame: pd.DataFrame, plate: Plate, ph: float | None) -> None:
    for row_index in range(len(frame)):
        if _text(frame.iat[row_index, 0]) != "Abs":
            continue
        wavelength = _wavelength_above(frame, row_index)
        data_row = row_index + 1
        while data_row < len(frame) and _ROW_LABEL.fullmatch(
            _text(frame.iat[data_row, 0])
        ):
            row_label = _text(frame.iat[data_row, 0])
            for column in range(1, frame.shape[1]):
                column_number = _as_float(frame.iat[row_index, column])
                absorption = _as_float(frame.iat[data_row, column])
                if column_number is None or absorption is None:
                    continue
                well_id = f"{row_label}{int(column_number)}"
                _add_measurement(
                    plate,
                    well_id,
                    ph,
                    wavelength,
                    [absorption],
                    [0.0],
                )
            data_row += 1


def _parse_kinetic(frame: pd.DataFrame, plate: Plate, ph: float | None) -> None:
    for row_index in range(len(frame)):
        if _text(frame.iat[row_index, 0]) != "Reading":
            continue
        wavelength = _wavelength_above(frame, row_index)
        headers = [_text(frame.iat[row_index, column]) for column in range(frame.shape[1])]
        time_column = next(
            (
                column
                for column, header in enumerate(headers)
                if header.lower().startswith("avg. time")
            ),
            None,
        )
        if time_column is None:
            raise ValueError("Kinetic block has no 'avg. time' column.")

        well_columns = _well_columns(headers)
        if not well_columns:
            raise ValueError("Kinetic block has no well columns.")

        times: list[float] = []
        values = {well_id: [] for _, well_id in well_columns}
        data_row = row_index + 1
        while data_row < len(frame) and _as_float(frame.iat[data_row, 0]) is not None:
            time = _as_float(frame.iat[data_row, time_column])
            if time is None:
                raise ValueError(f"Missing kinetic time in row {data_row}.")
            times.append(time)
            for column, well_id in well_columns:
                absorption = _as_float(frame.iat[data_row, column])
                if absorption is None:
                    raise ValueError(
                        f"Missing absorbance for well {well_id} in row {data_row}."
                    )
                values[well_id].append(absorption)
            data_row += 1

        for well_id, absorption in values.items():
            _add_measurement(plate, well_id, ph, wavelength, absorption, times)


def _parse_spectrum(frame: pd.DataFrame, plate: Plate, ph: float | None) -> None:
    header_rows = [
        row_index
        for row_index in range(len(frame))
        if _text(frame.iat[row_index, 0]) == "Wavelength"
    ]
    if not header_rows:
        raise ValueError("Spectrum sheet has no Wavelength column.")

    for row_index in header_rows:
        headers = [
            _text(frame.iat[row_index, column]) for column in range(frame.shape[1])
        ]
        well_columns = _well_columns(headers)
        if not well_columns:
            raise ValueError("Spectrum block has no well columns.")

        series: dict[str, list[tuple[float, float]]] = {
            well_id: [] for _, well_id in well_columns
        }
        data_row = row_index + 1
        while data_row < len(frame):
            wavelength = _as_float(frame.iat[data_row, 0])
            if wavelength is None:
                break
            for column, well_id in well_columns:
                absorption = _as_float(frame.iat[data_row, column])
                if absorption is None:
                    raise ValueError(
                        f"Missing spectrum value for well {well_id} at {wavelength} nm."
                    )
                series[well_id].append((wavelength, absorption))
            data_row += 1

        for well_id, points in series.items():
            well = _well(plate, well_id, ph)
            for wavelength, absorption in points:
                well.add_to_measurements(
                    wavelength=wavelength,
                    absorption=[absorption],
                    time=[0.0],
                    time_unit="s",
                )


def _add_measurement(
    plate: Plate,
    well_id: str,
    ph: float | None,
    wavelength: float,
    absorption: list[float],
    time: list[float],
) -> None:
    _well(plate, well_id, ph).add_to_measurements(
        wavelength=wavelength,
        absorption=absorption,
        time=time,
        time_unit="s",
    )


def _well(plate: Plate, well_id: str, ph: float | None) -> Well:
    for well in plate.wells:
        if well.id == well_id:
            return well
    x_pos, y_pos = id_to_xy(well_id)
    return plate.add_to_wells(id=well_id, x_pos=x_pos, y_pos=y_pos, ph=ph)


def _well_columns(headers: list[str]) -> list[tuple[int, str]]:
    columns = []
    for column, header in enumerate(headers):
        match = _WELL_HEADER.search(header)
        if match is None:
            continue
        well_id = f"{match.group(1).upper()}{int(match.group(2))}"
        columns.append((column, well_id))
    return columns


def _wavelength_above(frame: pd.DataFrame, row_index: int) -> float:
    for index in range(row_index, -1, -1):
        match = _WAVELENGTH_LINE.search(_text(frame.iat[index, 0]))
        if match is not None:
            return float(match.group(1).replace(",", "."))
    raise ValueError(f"No wavelength found above row {row_index}.")


def _detect_layout(frame: pd.DataFrame) -> str:
    first_column = [_text(frame.iat[row, 0]) for row in range(len(frame))]
    if "Reading" in first_column:
        return "kinetic"
    if "Wavelength" in first_column:
        return "spectrum"
    if "Abs" in first_column:
        return "endpoint"
    raise ValueError("Unrecognized Varioskan absorbance layout.")


def _find_sheet(workbook: pd.ExcelFile, sheet_name: str) -> str | None:
    for name in workbook.sheet_names:
        if name.lower() == sheet_name.lower():
            return name
    return None


def _first_absorbance_frame(workbook: pd.ExcelFile, path: str) -> pd.DataFrame:
    for sheet_name in workbook.sheet_names:
        if sheet_name.startswith("Absorbance"):
            return pd.read_excel(path, sheet_name=sheet_name, header=None)
    raise ValueError(f"No absorbance sheet found in {path}.")


def _value_beside_label(frame: pd.DataFrame, label: str) -> str | None:
    for row_index in range(len(frame)):
        for column in range(frame.shape[1]):
            if _text(frame.iat[row_index, column]) != label:
                continue
            for value_column in range(column + 1, frame.shape[1]):
                value = _text(frame.iat[row_index, value_column])
                if value:
                    return value
    return None


def _text(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _as_float(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing)
