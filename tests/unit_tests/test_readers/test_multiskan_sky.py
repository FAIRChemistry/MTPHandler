import pytest

from mtphandler.model import Plate
from mtphandler.readers import read_multiskan_sky

ph = 6.9


def test_read_multiskan_spectrum_1500():
    # Arrange
    path = "docs/examples/data/Multiskan Sky.xlsx"

    # Act
    plate = read_multiskan_sky(
        path=path,
        ph=ph,
    )

    # Assert
    assert isinstance(plate, Plate)
    assert plate.temperature_unit.name == "C"
    assert len(plate.wells) == 7

    for well in plate.wells:
        if well.id == "B2":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 1
            assert well.y_pos == 1
            measurment = well.measurements[0]
            assert measurment.wavelength == 340.0
            assert measurment.time_unit.name == "s"
            assert measurment.absorption[4] == pytest.approx(1.12, rel=1e-2)
            assert len(measurment.absorption) == 101

        if well.id == "G2":
            assert well.measurements[0].absorption[-1] == pytest.approx(
                0.1093, rel=1e-2
            )
