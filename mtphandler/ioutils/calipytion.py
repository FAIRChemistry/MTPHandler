from __future__ import annotations

from calipytion import Calibrator
from calipytion.model import Calibration, Sample

from mtphandler.model import Plate, Well
from mtphandler.molecule import Molecule
from mtphandler.tools import (
    get_measurement,
    get_species_condition,
    measurement_is_blanked_for,
    well_contains_species,
)


def _get_standard_wells(
    plate: Plate,
    protein_ids: list[str],
    molecule: Molecule,
    wavelength: float,
    silent: bool = False,
) -> list[Well]:
    """Goes through the wells and finds suitable standard wells.

    Args:
        plate (Plate): Plate with the wells.
        protein_ids (list[str]): IDs of the proteins that catalyze the reaction.
        molecule (Molecule): Molecule to calibrate.
        wavelength (float): Wavelength of the measurements.
        silent (bool, optional): If True, no print statements are shown. Defaults to False.

    Returns:
        list[Well]: List of wells that can be used as standards
    """
    # Subset of wells, that contain specified species, do not contain a protein, and are blanked

    # get wells with only one component, that does not contribute to the signal
    buffer_blank_wells = []
    standard_wells = []
    for well in plate.wells:
        measurement = get_measurement(well, wavelength)

        # get all wells with one init condition that has a concentration grater than 0
        # int_concs_creater_than_zero = [
        #     condition for condition in well.init_conditions if condition.init_conc > 0
        # ]

        # if len(int_concs_creater_than_zero) == 1:
        #     buffer_blank_wells.append(well)

        if not well_contains_species(well, molecule.id, conc_above_zero=True):
            continue

        if any(
            [
                well_contains_species(well, catalyst_id, conc_above_zero=True)
                for catalyst_id in protein_ids
            ]
        ):
            continue

        if measurement_is_blanked_for(measurement, molecule.id):
            standard_wells.append(well)

        # Add wells with zero concentration to standard wells
        if all(
            [
                blank_state.contributes_to_signal is False
                for blank_state in measurement.blank_states
            ]
        ):
            standard_wells.append(well)

    if not silent:
        print(
            f"🔎 Found {len(standard_wells)} wells containing {molecule.name} ({molecule.id})."
        )

    return standard_wells + buffer_blank_wells


def map_to_standard(
    plate: Plate,
    molecule: Molecule,
    protein_ids: list[str],
    wavelength: float,
) -> Calibration:
    standard_wells = _get_standard_wells(
        plate=plate,
        protein_ids=protein_ids,
        molecule=molecule,
        wavelength=wavelength,
    )
    print([well.id for well in standard_wells])

    # Map wells to samples of a standard
    samples = []
    phs = []
    for well in standard_wells:
        condition = get_species_condition(well, molecule.id)
        measurement = get_measurement(well, wavelength)

        samples.append(
            Sample(
                id=well.id,
                concentration=condition.init_conc,
                conc_unit=condition.conc_unit.name,
                signal=float(measurement.absorption[0]),
            )
        )
        phs.append(well.ph)

    # Check if all samples have the same pH
    if not all([ph == phs[0] for ph in phs]):
        raise ValueError(
            f"Samples of standard {molecule.name} have different pH values: {phs}"
        )
    ph = phs[0]

    # print lowest and highest signal
    print(
        min([sample.signal for sample in samples]),
        max([sample.signal for sample in samples]),
        "lowest and highest signal",
    )

    # Create standard
    return Calibration(
        molecule_id=molecule.id,
        molecule_symbol=molecule.id,
        pubchem_cid=molecule.pubchem_cid,
        molecule_name=molecule.name,
        wavelength=wavelength,
        samples=samples,
        ph=ph,
        temperature=plate.temperatures[0],
        temp_unit=plate.temperature_unit.name,
    )


def initialize_calibrator(
    plate: Plate,
    wavelength: float,
    molecule: Molecule,
    protein_ids: list[str],
    cutoff: float | None = None,
) -> Calibrator:
    """
    Initialize a calibrator for a given species.

    Args:
        plate (Plate): Plate with the wells.
        wavelength (float): Wavelength of the measurements.
        molecule (Molecule): Molecule to calibrate.
        protein_ids (list[str]): IDs of the proteins that catalyze the reaction.
        cutoff (float | None): Cutoff for the calibration. Calibration samples with
            a signal above the cutoff are ignored.
    """
    standard = map_to_standard(
        plate=plate,
        protein_ids=protein_ids,
        molecule=molecule,
        wavelength=wavelength,
    )

    return Calibrator.from_standard(standard, cutoff=cutoff)
