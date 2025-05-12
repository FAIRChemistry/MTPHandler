import numpy as np
import pytest

from mtphandler import PlateManager
from mtphandler.model import Plate
from mtphandler.molecule import Molecule


# create an artificaial plate with some wells and species
@pytest.fixture
def get_molecule():
    return Molecule(
        id="m1",
        pubchem_cid=12345,
        name="Test Molecule",
        constant=False,
    )


@pytest.fixture
def get_other_molecule():
    return Molecule(
        id="m2",
        pubchem_cid=123456,
        name="Test Molecule 2",
        constant=False,
    )


@pytest.fixture
def setup_handler() -> tuple[PlateManager, Plate]:
    plate = Plate(
        id="ID123",
        name="Test Plate",
        date_measured="2021-01-01 12:00:00",
        temperature_unit="C",
        temperatures=[37.0] * 12,
        time_unit="s",
        times=np.arange(0, 12, 1).tolist(),
    )
    w1 = plate.add_to_wells(
        id="A1",
        x_pos=0,
        y_pos=0,
        ph=7.0,
        volume=100,
        volume_unit="ul",
    )

    w2 = plate.add_to_wells(
        id="B1",
        x_pos=0,
        y_pos=1,
        ph=7.7,
        volume=200,
        volume_unit="ul",
    )

    w3 = plate.add_to_wells(
        id="A9",
        x_pos=8,
        y_pos=0,
        ph=8.0,
        volume=300,
        volume_unit="ul",
    )

    wells = [w1, w2, w3]

    for well in wells:
        well.add_to_measurements(
            wavelength=600,
            absorption=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            time_unit="s",
            time=np.arange(0, 12, 1).tolist(),
        )

    handler = PlateManager(plate=plate, name="Test Plate")

    return handler, plate


def test_add_well(setup_handler):
    handler, plate = setup_handler

    # Test adding a new well
    well = handler.plate.add_to_wells(
        id="C3", x_pos=2, y_pos=2, ph=7.5, volume=400, volume_unit="ul"
    )
    assert well.id == "C3"
    assert well.x_pos == 2
    assert well.y_pos == 2
    assert well.ph == 7.5
    assert well.volume == 400
    assert well.volume_unit.name == "ul"
    assert len(plate.wells) == 4

    # test adding measurements to the well
    well.add_to_measurements(
        wavelength=600,
        absorption=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        time_unit="s",
        time=np.arange(0, 12, 1).tolist(),
    )
    assert len(well.measurements) == 1
    assert well.measurements[0].wavelength == 600
    assert well.measurements[0].time_unit.name == "s"
    assert well.measurements[0].time == np.arange(0, 12, 1).tolist()
    assert well.measurements[0].absorption == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
    ]


def test_assing_to_all_wells(setup_handler, get_molecule, get_other_molecule):
    handler, plate = setup_handler

    mol1 = get_molecule
    mol2 = get_other_molecule

    # Test assigning a species to all wells
    handler.add_molecule(molecule=mol1)
    handler.add_molecule(molecule=mol2)

    handler._assign_to_all(
        species=mol1,
        init_conc=1.2,
        conc_unit="mmol / l",
        contributes_to_signal=None,
        silent=False,
    )
    for well in plate.wells:
        assert well.measurements[0].blank_states[0].species_id == mol1.id
        assert well.measurements[0].blank_states[0].contributes_to_signal

        assert well.init_conditions[0].species_id == mol1.id
        assert well.init_conditions[0].init_conc == 1.2

    handler._assign_to_all(
        species=mol2,
        init_conc=0,
        conc_unit="mmol / l",
        contributes_to_signal=None,
        silent=False,
    )

    for well in plate.wells:
        assert well.measurements[0].blank_states[1].species_id == mol2.id
        assert not well.measurements[0].blank_states[1].contributes_to_signal

        assert well.init_conditions[1].species_id == mol2.id
        assert well.init_conditions[1].init_conc == 0


def test_assign_to_columns(setup_handler, get_molecule, get_other_molecule):
    handler, plate = setup_handler

    mol1 = get_molecule
    mol2 = get_other_molecule

    # Test assigning a species to all wells
    handler.add_molecule(molecule=mol1)
    handler.add_molecule(molecule=mol2)

    handler._assign_to_columns(
        species=mol1,
        init_concs=[1, 2],
        conc_unit="mmol / l",
        contributes_to_signal=None,
        column_ids=[1],
        silent=False,
    )

    for well in plate.wells:
        if well.x_pos == 0:
            assert well.measurements[0].blank_states[0].species_id == mol1.id
            assert well.measurements[0].blank_states[0].contributes_to_signal

            assert well.init_conditions[0].species_id == mol1.id
        elif well.id == "A1":
            assert well.init_conditions[0].init_conc == 1
        elif well.id == "A2":
            assert well.init_conditions[0].init_conc == 2

        else:
            assert len(well.measurements[0].blank_states) == 0
            assert len(well.init_conditions) == 0

    handler._assign_to_columns(
        species=mol2,
        init_concs=[12],
        conc_unit="mmol / l",
        contributes_to_signal=None,
        column_ids=[9],
        silent=False,
    )

    for well in plate.wells:
        if well.id == "A9":
            assert well.measurements[0].blank_states[0].species_id == mol2.id
            assert well.measurements[0].blank_states[0].contributes_to_signal

            assert well.init_conditions[0].species_id == mol2.id
            assert well.init_conditions[0].init_conc == 12

    # test number of wells in column and init concs does not match
    with pytest.raises(AssertionError):
        handler._assign_to_columns(
            species=mol1,
            init_concs=[1, 2],
            conc_unit="mmol / l",
            contributes_to_signal=None,
            column_ids=[2],
            silent=False,
        )


def test_add_molecule_without_constant(setup_handler):
    plate_manager, plate = setup_handler

    molecule = Molecule(
        id="m1", pubchem_cid=12345, name="Test Molecule", constant=False
    )

    plate_manager.add_molecule(molecule)

    assert len(plate_manager.molecules) == 1
    assert plate_manager.molecules[0].id == "m1"
    assert plate_manager.molecules[0].constant == False


def test_update_existing_molecule(setup_handler):
    plate_manager, plate = setup_handler

    molecule1 = Molecule(
        id="m1", pubchem_cid=12345, name="Test Molecule", constant=False
    )
    molecule2 = Molecule(
        id="m1", pubchem_cid=12345, name="Updated Molecule", constant=True
    )

    plate_manager.add_molecule(molecule1)
    plate_manager.add_molecule(molecule2)

    assert len(plate_manager.molecules) == 1
    assert plate_manager.molecules[0].id == "m1"
    assert plate_manager.molecules[0].name == "Updated Molecule"
    assert plate_manager.molecules[0].constant == True


def test_define_molecule_with_valid_data(setup_handler):
    plate_manager, plate = setup_handler

    molecule = plate_manager.define_molecule(
        id="m1", pubchem_cid=12345, name="Test Molecule", constant=False
    )

    assert molecule.id == "m1"
    assert molecule.pubchem_cid == 12345
    assert molecule.name == "Test Molecule"
    assert molecule.constant is False


def test_define_molecule_with_non_integer_pubchem_cid(setup_handler):
    plate_manager, plate = setup_handler

    with pytest.raises(ValueError, match="PubChem CID must be an integer."):
        plate_manager.define_molecule(
            id="m1",
            pubchem_cid="not_an_integer",
            name="Test Molecule",
            constant=False,
        )


def test_define_molecule_without_name(setup_handler):
    # Initialize the class object
    plate_manager, plate = setup_handler

    # Define a molecule with valid id and pubchem_cid, but without name
    molecule_id = "m1"
    pubchem_cid = 12345

    mol = plate_manager.define_molecule(id=molecule_id, pubchem_cid=pubchem_cid)

    assert len(mol.name) > 0


def test_append_molecule_if_not_exist(setup_handler):
    # Initialize the PlateManager object
    plate_manager, plate = setup_handler

    # Define a molecule with PubChem CID not available
    molecule = plate_manager.define_molecule(
        id="m1", pubchem_cid=-1, name="Test Molecule", constant=False
    )

    # Check if the molecule is correctly appended to the list
    assert len(plate_manager.molecules) == 1
    assert plate_manager.molecules[0].id == "m1"
    assert plate_manager.molecules[0].name == "Test Molecule"
    assert plate_manager.molecules[0].constant == False


def test_defining_already_defined_molecule_updates_molecule(setup_handler):
    # Initialize the PlateManager object
    plate_manager, plate = setup_handler

    # Define a molecule with PubChem CID not available
    molecule = plate_manager.define_molecule(
        id="m1", pubchem_cid=-1, name="Test Molecule", constant=False
    )

    # Check if the molecule is correctly appended to the list
    assert len(plate_manager.molecules) == 1
    assert plate_manager.molecules[0].id == "m1"
    assert plate_manager.molecules[0].name == "Test Molecule"
    assert plate_manager.molecules[0].constant == False

    # Define the molecule again with different properties
    molecule = plate_manager.define_molecule(
        id="m1", pubchem_cid=12345, name="New Molecule", constant=True
    )

    # Check if the molecule is updated
    assert len(plate_manager.molecules) == 1
    assert plate_manager.molecules[0].id == "m1"
    assert plate_manager.molecules[0].name == "New Molecule"
    assert plate_manager.molecules[0].constant == True


def test_define_molecule_pubchem_cid_minus_one_no_name(setup_handler):
    # Initialize the class object
    plate_manager, plate = setup_handler

    # Define molecule with pubchem_cid=-1 and no name provided
    with pytest.raises(ValueError):
        molecule = plate_manager.define_molecule(id="test_id", pubchem_cid=-1)


def test_molecule_string_pubchem_cid(setup_handler):
    # Initialize the class object
    plate_manager, plate = setup_handler

    # Define molecule with invalid pubchem_cid
    with pytest.raises(ValueError, match="PubChem CID must be an integer."):
        molecule = plate_manager.define_molecule(
            id="test_id", pubchem_cid="invalid_pubchem_cid"
        )


def test_molecule_cid_not_exist(setup_handler):
    # Initialize the class object
    plate_manager, plate = setup_handler

    # Define molecule with pubchem_cid=-1 and no name provided
    with pytest.raises(
        ValueError, match="Failed to retrieve molecule name from PubChem"
    ):
        molecule = plate_manager.define_molecule(
            id="test_id", pubchem_cid=12312323235235235235
        )
