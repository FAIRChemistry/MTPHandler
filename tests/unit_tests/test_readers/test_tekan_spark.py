import pytest

from mtphandler.model import Plate
from mtphandler.readers import read_tekan_spark

ph = 6.9


def test_read_tekan_spark_kinetic():
    """Test reading kinetic (time series) Tekan Spark data."""
    # Arrange
    path = "docs/examples/data/tekan_spark.xlsx"

    # Act
    plate = read_tekan_spark(path=path, ph=ph)

    # Assert
    assert isinstance(plate, Plate)
    assert plate.temperature_unit.name == "C"
    assert len(plate.wells) == 45
    assert plate.temperatures[0] == pytest.approx(24.7, rel=1e-2)
    assert len(plate.times) == 61  # Multiple timepoints for kinetic data
    assert plate.times[0] == 0.0  # First timepoint
    assert plate.times[-1] == pytest.approx(30.0, rel=1e-2)  # Last timepoint

    for well in plate.wells:
        if well.id == "A1":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 0
            assert well.y_pos == 0
            measurement = well.measurements[0]
            assert measurement.wavelength == 420.0  # Absorbance wavelength
            assert measurement.absorption[4] == pytest.approx(1.0332, rel=1e-3)
            assert len(measurement.absorption) == 61  # Time series data

        if well.id == "C1":
            assert len(well.measurements) == 1
            assert well.ph == ph
            assert well.x_pos == 0
            assert well.y_pos == 2
            measurement = well.measurements[0]
            assert measurement.wavelength == 420.0  # Absorbance wavelength
            assert measurement.absorption[-1] == pytest.approx(0.0862, rel=1e-3)
            assert len(measurement.absorption) == 61  # Time series data


def test_read_tekan_spark_endpoint():
    """Test reading endpoint (single timepoint) Tekan Spark data."""
    # Arrange
    path = "docs/examples/data/tekan_spark_endpoint.xlsx"

    # Act
    plate = read_tekan_spark(path=path, ph=ph)

    # Assert
    assert isinstance(plate, Plate)
    assert plate.temperature_unit.name == "C"
    assert len(plate.wells) == 9  # Only wells with data
    assert len(plate.temperatures) == 1  # Single temperature for endpoint
    assert plate.temperatures[0] == 25.5  # Temperature from metadata
    assert len(plate.times) == 1  # Single timepoint for endpoint
    assert plate.times[0] == 0.0  # Single timepoint

    # Check specific wells with known data
    well_c11 = next(w for w in plate.wells if w.id == "C11")
    assert well_c11.ph == ph
    assert well_c11.x_pos == 10  # Column 11 -> index 10
    assert well_c11.y_pos == 2  # Row C -> index 2
    assert len(well_c11.measurements) == 1
    measurement = well_c11.measurements[0]
    assert measurement.wavelength == 0.0  # Luminescence (no wavelength)
    assert measurement.absorption[0] == 47088.0  # Expected value
    assert len(measurement.absorption) == 1  # Single measurement

    well_d12 = next(w for w in plate.wells if w.id == "D12")
    assert well_d12.ph == ph
    assert well_d12.x_pos == 11  # Column 12 -> index 11
    assert well_d12.y_pos == 3  # Row D -> index 3
    measurement = well_d12.measurements[0]
    assert measurement.wavelength == 0.0  # Luminescence
    assert measurement.absorption[0] == 128430.0  # Expected value
    assert len(measurement.absorption) == 1  # Single measurement


def test_read_tekan_spark_temperature_validation():
    """Test that temperature validation works correctly."""
    # This would need a modified test file without temperature data
    # For now, we just verify that our real files have proper temperature data

    # Kinetic data should have temperature series
    kinetic_plate = read_tekan_spark("docs/examples/data/tekan_spark.xlsx", ph=7.4)
    assert len(kinetic_plate.temperatures) > 1
    assert all(
        t > 0 for t in kinetic_plate.temperatures
    )  # All temps should be positive

    # Endpoint data should have single temperature from metadata
    endpoint_plate = read_tekan_spark(
        "docs/examples/data/tekan_spark_endpoint.xlsx", ph=7.4
    )
    assert len(endpoint_plate.temperatures) == 1
    assert endpoint_plate.temperatures[0] == 25.5


def test_read_tekan_spark_wavelength_validation():
    """Test that wavelength detection works correctly for both data types."""
    # Kinetic data should have proper wavelength from metadata
    kinetic_plate = read_tekan_spark("docs/examples/data/tekan_spark.xlsx", ph=7.4)
    assert kinetic_plate.wells[0].measurements[0].wavelength == 420.0

    # Endpoint data should detect luminescence (wavelength = 0)
    endpoint_plate = read_tekan_spark(
        "docs/examples/data/tekan_spark_endpoint.xlsx", ph=7.4
    )
    assert endpoint_plate.wells[0].measurements[0].wavelength == 0.0


def test_read_tekan_spark_data_integrity():
    """Test data integrity and structure for both data types."""
    kinetic_plate = read_tekan_spark("docs/examples/data/tekan_spark.xlsx", ph=7.4)
    endpoint_plate = read_tekan_spark(
        "docs/examples/data/tekan_spark_endpoint.xlsx", ph=7.4
    )

    # Kinetic data validation
    assert len(kinetic_plate.wells) == 45  # Full plate
    assert len(kinetic_plate.times) == 61  # Multiple timepoints
    assert all(len(w.measurements[0].absorption) == 61 for w in kinetic_plate.wells)

    # Endpoint data validation
    assert len(endpoint_plate.wells) == 9  # Only wells with data
    assert len(endpoint_plate.times) == 1  # Single timepoint
    assert all(len(w.measurements[0].absorption) == 1 for w in endpoint_plate.wells)

    # Check that all wells have proper coordinates
    for plate in [kinetic_plate, endpoint_plate]:
        for well in plate.wells:
            assert well.x_pos >= 0
            assert well.y_pos >= 0
            assert len(well.measurements) == 1
            assert well.ph == 7.4
