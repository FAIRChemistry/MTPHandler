from MTPHandler.core import Plate
from MTPHandler.readers.megellan_parser import read_magellan


def test_magellan():
    # Arrange
    cls = Plate
    path = "tests/magellan.xlsx"
    ph = 7.0
    wavelength = 450.0

    # Act
    plate = read_magellan(cls, path, ph, wavelength)

    # Assert
    assert isinstance(plate, Plate)
    assert plate.n_rows == 8
    assert plate.n_columns == 12
    assert plate.measured_wavelengths == [450.0]
    assert plate.temperature_unit == "C"
    assert plate.temperatures == [37.1, 37.2, 37.1, 37.2, 36.9, 37.3, 37.2]
    assert plate.time_unit == "s"
    assert plate.times == [0, 900.0, 1802.0, 2700.0, 3600.0, 4502.0, 5400.0]
    assert len(plate.wells) == 62
    assert plate.get_well("E8").get_measurement(450.0).absorptions[0] == 2.6871
