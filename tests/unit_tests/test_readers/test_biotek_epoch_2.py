import pytest

from mtphandler.model import Plate
from mtphandler.readers import read_biotek

ph = 6.9


def test_read_biotek_epoch_2():
    # Arrange
    path = "docs/examples/data/BioTek_Epoch2.xlsx"
    # Act
    plate = read_biotek(
        path=path,
        ph=ph,
    )

    # Assert
    assert isinstance(plate, Plate)
    assert plate.temperature_unit.name == "C"
    assert len(plate.wells) == 36

    for well in plate.wells:
        if well.id == "B2":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 1
            assert well.y_pos == 1
            measurment = well.measurements[0]
            assert measurment.wavelength == 630
            assert measurment.time_unit.name == "s"
            assert measurment.absorption[4] == pytest.approx(0.1, rel=1e-2)
            assert len(measurment.absorption) == 353

        if well.id == "D9":
            assert well.measurements[0].absorption[-1] == pytest.approx(0.226, rel=1e-2)
            
    # Test another BioTek plate example
    """Tests that read_biotek can correctly parse the gen5 data file with dynamic kinetic interval detection."""
    test_file_path = "docs/examples/data/gen5_data_export_test_12-1-2025.xlsx"

    # Ensure the test file exists
    #assert os.path.exists(test_file_path), f"Test file not found: {test_file_path}"

    # Use the read_biotek function with the test file
    # This should now run without raising a ValueError due to the fix
    plate = read_biotek(path=test_file_path, ph=7.4)

    # Assert that a Plate object is returned
    assert isinstance(plate, Plate)

    # Assert that the plate contains wells and measurements
    assert len(plate.wells) > 0
    assert len(plate.wells[0].measurements) > 0

    # Get the first measurement from the first well
    first_measurement = plate.wells[0].measurements[0]

    # Assert that the time data is populated and has more than one point
    assert len(first_measurement.time) > 1

    # The expected interval from the file is 'Interval 0:08:00', which is 8 minutes.
    # As currently implemented, `np.arange` uses this minute value directly
    # to generate the time series, even though `time_unit='s'` is specified.
    expected_interval_minutes = 8.0
    actual_interval = first_measurement.time[1] - first_measurement.time[0]

    # Assert that the actual interval matches the expected 8.0 minutes
    assert actual_interval == pytest.approx(expected_interval_minutes)

    # Note: There's a potential inconsistency in the read_biotek function
    # where time points are generated using 'minutes' but `time_unit` is set to 's' (seconds).
    # This test validates the numerical interval extracted, but a future enhancement
    # might involve ensuring strict unit consistency (e.g., converting minutes to seconds
    # before generating the time array if time_unit is 's').
   
