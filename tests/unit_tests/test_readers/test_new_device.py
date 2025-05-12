from mtphandler.readers import read_new_device


def test_new_reader_kinetic():
    path = "docs/examples/data/new_device_kinetic.xlsx"

    plate = read_new_device(
        path=path,
        ph=7.4,
        temperature=[37.0],
        temperature_unit="C",
    )

    assert len(plate.wells) == 72
    assert plate.wells[0].x_pos == 0
    assert plate.wells[0].y_pos == 0
    assert plate.wells[0].id == "ArODH0001"
    assert plate.wells[-1].x_pos == 11
    assert plate.wells[-1].y_pos == 5
    assert plate.wells[-1].id == "mODH-582"
    assert plate.wells[-1].measurements[0].absorption[0] == 0.5626
    assert plate.wells[-1].measurements[0].absorption[-1] == 0.6376


def test_new_reader_endpoint():
    path = "docs/examples/data/new_device_calib.xlsx"

    plate = read_new_device(
        path=path,
        ph=7.4,
        temperature=[37.0],
        temperature_unit="C",
    )

    assert len(plate.wells) == 42
    assert plate.wells[0].x_pos == 0
    assert plate.wells[0].y_pos == 0
    assert plate.wells[0].id == "NADH0001"
    assert plate.wells[0].measurements[0].absorption[0] == 3.116
    assert plate.name == "NADH_NADPH_calib.skax"
    assert plate.date_measured == "2023-08-08 04:06:03"
