import re
from io import StringIO

import numpy as np
import pandas as pd
from loguru import logger

from mtphandler.model import Plate, Well
from mtphandler.readers.utils import id_to_xy, xy_to_id


class WrongParserError(Exception):
    """Exception raised when the wrong parser is used to read a file."""

    def __init__(self, parser_name, expected_format):
        self.parser_name = parser_name
        self.expected_format = expected_format
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (
            f"Error in {self.parser_name}: Expected format '{self.expected_format}', "
        )


def read_spectra_max_190(path, ph: float | None) -> Plate:
    """
    Reads SpectraMax 190 data from a file and returns a plate object.

    Args:
        path (str): The path to the file containing the SpectraMax 190 data.
        ph (float | None, optional): The pH value. Defaults to None.

    Returns:
        plate: The plate object containing the data.

    Raises:
        WrongParserError: If the file format is not SpectraMax 190.
        ValueError: If the wavelengths could not be extracted or the data blocks are not of equal shape.

    """

    iso_encoding = "ISO-8859-1"
    utf16_encoding = "utf-16"

    try:
        lines = open_file(path, iso_encoding)
        if "##BLOCKS" not in lines[0]:
            raise ValueError
    except ValueError:
        lines = open_file(path, utf16_encoding)
        if "##BLOCKS" not in lines[0]:
            raise WrongParserError(
                parser_name="read_spectra_max_190",
                expected_format="SpectraMax 190",
            )

    wavelength_pattern = r"(?:[^\t]*\t){15}(\d+)"

    # Extract the wavelength
    try:
        wavelength_line = [line for line in lines if line.startswith("Plate:")][0]
        wavelength = float(re.findall(wavelength_pattern, wavelength_line)[0])
    except ValueError:
        raise ValueError("Wavelengths could not be extracted.")

    try:
        blocks = identify_blocks(lines)

        times, temperatures, blocks = sanitize_blocks(blocks)

        try:
            data_matrix = np.array(blocks)
            data_matrix = data_matrix.swapaxes(0, 2)
        except ValueError:
            raise ValueError(
                "Data blocks are not of equal shape, file seems corrupted."
            )
        plate = map_to_plate(data_matrix, times, temperatures, ph, wavelength)

        return plate

    except IndexError:
        for line_id, line in enumerate(lines):
            if line.startswith("Time"):
                start_id = line_id
            if line.startswith("\n"):
                end_id = line_id
                break

        data = lines[start_id:end_id]
        # make pandas df from öist of strings
        data_str = "\n".join(data)

        # Use StringIO to simulate a file object
        data_io = StringIO(data_str)

        # Use pd.read_csv with sep='\t' to read the data into a DataFrame
        df = pd.read_csv(data_io, sep="\t")
        # drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        time = df.pop("Time")
        time = [time_to_min_float(t) for t in time]
        temperatures = df.pop("Temperature(¡C)").values.tolist()

        # iterate over the columns and create wells
        wells = []
        for column in df.columns:
            x, y = id_to_xy(column)
            well = Well(
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
            wells.append(well)

        plate = Plate(
            time_unit="s",
            temperatures=temperatures,
            temperature_unit="C",
            wells=wells,
        )

    return plate


def map_to_plate(
    data_matrix: np.ndarray,
    times: list[float],
    temperatures: list[float],
    ph: float | None,
    wavelength: float,
):
    """
    Maps a data matrix to a Plate object.

    Args:
        data_matrix (np.ndarray): The data matrix containing the measurements.
        times (list[float]): The list of time values.
        temperatures (list[float]): The list of temperature values.
        ph (float | None): The pH value or None if not applicable.
        wavelength (float): The wavelength value.

    Returns:
        Plate: The Plate object containing the mapped data.
    """

    wells = []
    for column_id in range(data_matrix.shape[0]):
        for row_id in range(data_matrix.shape[1]):
            well = Well(
                id=xy_to_id(column_id, row_id),
                ph=ph,
                x_pos=column_id,
                y_pos=row_id,
            )
            assert len(times) == data_matrix[column_id, row_id].size, (
                "Time and data length mismatch."
            )
            well.add_to_measurements(
                wavelength=wavelength,
                absorption=data_matrix[column_id, row_id].tolist(),
                time=times,
                time_unit="s",
            )
            wells.append(well)

    # Create plate
    plate = Plate(
        time_unit="s",
        temperatures=temperatures,
        temperature_unit="C",
        wells=wells,
    )

    return plate


def identify_blocks(lines):
    """Identify blocks in the file."""
    blocks = []
    current_block = []
    time_pattern = re.compile(
        r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s|$)"
    )  # Pattern to match h:mm or hh:mm:ss format

    # get line number is which time pattern is found
    section_starts = np.array(
        [i for i, line in enumerate(lines) if time_pattern.match(line)]
    )

    # check if the distance between the time pattern is consistent
    if not np.all(np.diff(section_starts) == section_starts[1] - section_starts[0]):
        logger.debug("Inconsistent time pattern found in file.")
    # get the number of lines in each section
    section_length = section_starts[1] - section_starts[0]
    section_ends = section_starts + section_length - 1

    # build slices for each section for extraction into list of blocks
    slices = [slice(start, end) for start, end in zip(section_starts, section_ends)]

    for s in slices:
        current_block = [line for line in lines[s]]
        if len(current_block[0].split(",")) == 2:
            continue
        blocks.append(current_block)

    return blocks


def is_increasing_by_one(lst):
    return all(lst[i] + 1 == lst[i + 1] for i in range(len(lst) - 1))


def sanitize_blocks(blocks):
    times, temperatures = [], []

    for block_id, block in enumerate(blocks):
        # Filter out empty lines
        block = [
            line
            for line in block
            if line.replace("\t", "").replace("\n", "").replace("\r", "").strip() != ""
        ]

        if not block:
            continue  # skip the entire block if it's empty

        for line_id, line in enumerate(block):
            if line_id == 0:
                time, temp, line = line.split("\t", 2)
                times.append(time_to_min_float(time))
                temperatures.append(float(temp.replace(",", ".")))

            line = line.strip()
            line = line.replace(",", ".")
            line = [float(entry) for entry in line.split("\t") if entry != ""]

            block[line_id] = line

        blocks[block_id] = block

    return times, temperatures, blocks


def time_to_min_float(time_str: str):
    time_parts = time_str.split(":")
    # Calculate time since zero in minutes
    if len(time_parts) == 3:  # hh:mm:ss format
        h, m, s = time_parts
        return float(h) * 60 + float(m) + float(s) / 60
    elif len(time_parts) == 2:  # h:mm format
        m, s = time_parts
        return float(m) + float(s) / 60
    else:
        raise ValueError(f"Unexpected time format: '{time_str}'")


def open_file(path: str, encoding: str):
    with open(path, "r", encoding=encoding) as file:
        lines = file.readlines()
        return lines
