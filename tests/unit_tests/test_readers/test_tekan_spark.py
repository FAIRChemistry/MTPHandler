import pytest

from mtphandler.model import Plate
from mtphandler.readers import read_tekan_spark

ph = 6.9


def test_read_tekan_spark():
    # Arrange
    path = "docs/examples/data/tekan_spark.xlsx"
    # Act
    plate = read_tekan_spark(
        path=path,
        ph=ph,
    )

    # Assert
    assert isinstance(plate, Plate)
    assert plate.temperature_unit.name == "C"
    assert len(plate.wells) == 45
    assert plate.temperatures[0] == pytest.approx(24.7, rel=1e-2)

    for well in plate.wells:
        if well.id == "A1":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 0
            assert well.y_pos == 0
            measurment = well.measurements[0]
            assert measurment.wavelength == 420.0
            assert measurment.absorption[4] == pytest.approx(1.0332, rel=1e-3)
            assert len(measurment.absorption) == 61

        if well.id == "C1":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 0
            assert well.y_pos == 2
            measurment = well.measurements[0]
            assert measurment.wavelength == 420.0
            assert measurment.absorption[-1] == pytest.approx(0.0862, rel=1e-3)
            assert len(measurment.absorption) == 61
