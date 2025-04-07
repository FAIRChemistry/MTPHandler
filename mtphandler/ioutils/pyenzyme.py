import numpy as np
import pyenzyme as pe
from calipytion.tools.calibrator import Calibrator
from loguru import logger
from pyenzyme.model import DataTypes
from pyenzyme.model import UnitDefinition as EnzML_UnitDef

from mtphandler.model import (
    InitCondition,
    PhotometricMeasurement,
    Plate,
    UnitDefinition,
    Well,
)
from mtphandler.molecule import Molecule, Protein


class Plate_to_EnzymeMLDocument:
    """Converts a Plate object along with associated molecules and proteins to an EnzymeMLDocument.
    If `to_concentration=True`, the absorption data is converted to concentration data using the calibrator.


    Raises:
        ValueError: If the pH of a well is not defined.
        ValueError: If no measurements were added to EnzymeML.

    Returns:
        EnzymeMLDocument: The EnzymeMLDocument containing the converted data.
    """

    def __init__(
        self,
        name: str,
        plate: Plate,
        well_ids: list[str] | None,
        molecules: list[Molecule],
        detected_molecule: Molecule,
        proteins: list[Protein],
        wavelength: float | None,
        catalyzed_only: bool,
        to_concentration: bool,
        extrapolate: bool,
        silent: bool,
    ) -> None:
        self.name = name
        self.plate = plate
        self.well_ids = well_ids
        self.molecules = molecules
        self.detected_molecule = detected_molecule
        self.proteins = proteins
        self.wavelength = wavelength
        self.calibrator_dict: dict[str, Calibrator] = {}
        self.catalyzed_only = catalyzed_only
        self.to_concentration = to_concentration
        self.extrapolate = extrapolate
        self.silent = silent

        # Initialize calibrators if concentration data is requested
        if self.to_concentration:
            self._init_calibrators()

        # Check if a wavelength was specified, otherwise set it to the only wavelength measured
        if self.wavelength is None:
            self._handle_wavelength()

    def convert(self):
        """
        Converts proteins, small molecules, and measurements to an EnzymeMLDocument.

        Returns:
            EnzymeMLDocument: The EnzymeMLDocument containing the converted data.
        """
        enzml_doc = pe.EnzymeMLDocument(name=self.name)
        logger.debug(f"Initialized EnzymeMLDocument with name {self.name}")

        dummy_vessel = self._get_dummy_vessel()
        enzml_doc.vessels = [dummy_vessel]

        # Add proteins to EnzymeML document
        enzml_doc.proteins = [
            self.map_protein(protein, dummy_vessel.id) for protein in self.proteins
        ]
        logger.debug(f"Added {len(self.proteins)} proteins to EnzymeMLDocument")

        # Add small molecules to EnzymeML document
        enzml_doc.small_molecules = [
            self.map_small_molecule(molecule, dummy_vessel.id)
            for molecule in self.molecules
        ]
        logger.debug(f"Added {len(self.molecules)} small molecules to EnzymeMLDocument")

        # Add measurements to EnzymeML document
        enzml_doc.measurements = self.wells_to_enzml_measurements()

        return enzml_doc

    def get_well_subset(self) -> list[Well]:
        """
        Returns a subset of wells from the plate based on the well ids.

        Returns:
            list[Well]: List of Well objects.
        """
        if self.well_ids is None:
            return self.plate.wells

        if isinstance(self.well_ids, str):
            self.well_ids = [self.well_ids]

        self.well_ids = [well_id.upper() for well_id in self.well_ids]
        subset = [well for well in self.plate.wells if well.id.upper() in self.well_ids]

        if len(subset) == 0:
            raise ValueError("No wells found with the specified well ids.")

        return subset

    def wells_to_enzml_measurements(self) -> list[pe.Measurement]:
        """Converts wells to EnzymeML measurements.

        Raises:
            ValueError: If the pH of a well is not defined.
            ValueError: If no measurements were added to EnzymeML.

        Returns:
            list[pe.Measurement]: List of EnzymeML `Measurement` objects.
        """
        meas_counter = 0
        measurements = []

        for well in self.get_well_subset():
            photo_measurement = next(
                (
                    meas
                    for meas in well.measurements
                    if meas.wavelength == self.wavelength
                ),
                None,
            )

            time_unit = well.measurements[0].time_unit

            # Skip wells without a measurement at the specified wavelength
            if not photo_measurement:
                continue

            # Skip wells without a protein if the flag is set
            if self.catalyzed_only:
                if not self.is_catalyzed(
                    well,
                    photo_measurement,
                    {p.id for p in self.proteins},
                    self.detected_molecule.id,
                ):
                    continue

            # Ensure that the pH of the well is defined
            if well.ph is None:
                raise ValueError(f"pH of well {well.id} is not defined.")

            # Create EnzymeML measurement
            enzml_meas = pe.Measurement(
                id=well.id,
                name="photometric measurement",
                ph=well.ph,
                group_id="",
                temperature=self.temperature,
                temperature_unit=EnzML_UnitDef(
                    **self.plate.temperature_unit.model_dump()
                ),
            )

            logger.debug(
                f"Contributing species in well {well.id}: {[(state.species_id, state.contributes_to_signal) for state in photo_measurement.blank_states]}"
            )

            # Check if only one species contributes to the signal
            measured_species = self.get_only_contributing_species(
                photo_measurement, well.id, self.detected_molecule.id
            )
            if not measured_species:
                continue

            logger.debug(
                f"Adding measurement from well {well.id} with species {measured_species}"
            )

            # Add species data to the measurement based on the initial conditions of the well
            self.add_to_species_data(enzml_meas, well.init_conditions, time_unit)

            # Add absorption data to the species data of the measurement
            self.add_absorption_data(
                measurement=enzml_meas,
                photo_measurement=photo_measurement,
                species_id=measured_species,
            )

            measurements.append(enzml_meas)

            meas_counter += 1

        if meas_counter == 0:
            raise ValueError("No measurements were added to EnzymeML.")

        if not self.silent:
            mode = "concentration" if self.to_concentration else "absorbance"
            print(
                f"✅ Added measurements from {meas_counter} wells with {mode} values to EnzymeMLDocument"
            )

        return measurements

    def add_absorption_data(
        self,
        measurement: pe.Measurement,
        photo_measurement: PhotometricMeasurement,
        species_id: str,
    ) -> None:
        """Adds absorption data to the species data of the measurement.
        Based in the `to_concentration` flag, the absorption data is converted to concentration data using the calibrator.

        Args:
            measurement (pe.Measurement): EnzymeML `Measurement` object.
            photo_measurement (PhotometricMeasurement): PhotometricMeasurement object.
            species_id (str): Species ID.

        Raises:
            ValueError: If the calibrator for the species is not defined.
        """

        species_data = next(
            (
                data
                for data in measurement.species_data
                if data.species_id == species_id
            ),
            None,
        )

        assert species_data is not None, (
            f"Species {species_id} not found in measurement {measurement.id}."
        )

        if self.to_concentration:
            data_type = pe.DataTypes.CONCENTRATION
            if species_id not in self.calibrator_dict:
                raise ValueError(
                    f"Calibrator for species {species_id} is not defined. Set `to_concentration=False`, or define a standard for species {species_id}."
                )

            data = self.calibrator_dict[species_id].calculate_concentrations(
                model=self.calibrator_dict[species_id].models[0],
                signals=photo_measurement.absorption,
                extrapolate=self.extrapolate,
            )
        else:
            data_type = pe.DataTypes.ABSORBANCE
            data = photo_measurement.absorption

        species_data.data_type = data_type
        species_data.data = data
        species_data.time = photo_measurement.time

        species_data.initial = species_data.data[0]

    @staticmethod
    def _get_dummy_vessel():
        from pyenzyme.units import dimensionless

        return pe.Vessel(
            id="v1",
            name="microtiter plate",
            volume=0,
            unit=dimensionless,
        )

    @property
    def temperature(self) -> float:
        return np.mean(self.plate.temperatures).tolist()

    @staticmethod
    def get_only_contributing_species(
        photo_measurement: PhotometricMeasurement,
        well_id: str,
        detected_molecule_id: str,
    ) -> str | None:
        # check that only one species contributes to the signal
        contributing_species = set()
        for state in photo_measurement.blank_states:
            if state.species_id == detected_molecule_id:
                contributing_species.add(state.species_id)
            if state.contributes_to_signal:
                contributing_species.add(state.species_id)

        if len(contributing_species) > 1:
            raise ValueError(
                f"""
                Multiple species ({contributing_species}) contribute to the signal in well {well_id}. Only one species is allowed."
                Either the plate was not blanked, or control measurements for determining the blank are missing.
                Species can manually be specified not to contribute to the signal by setting the `contributes_to_signal=False` during 
                the assignment of well conditions.
                """
            )

        if len(contributing_species) == 0:
            return None

        return contributing_species.pop()

    @staticmethod
    def is_catalyzed(
        well: Well,
        photo_measurement: PhotometricMeasurement,
        protein_ids: set[str],
        target_species: str,
    ) -> bool:
        """
        Checks if a well contains a catalyst and another species.

        Args:
            well (Well): `Well` object
            protein_ids (list[str]): List of protein ids
            target_species (str): Species ID of the target species which should be detected.

        Returns:
            bool: True if the well contains a catalyst and another species, False otherwise
        """

        contains_protein = False
        for condition in well.init_conditions:
            if condition.species_id in protein_ids and condition.init_conc > 0:
                contains_protein = True

        protein_contributes = False
        for state in photo_measurement.blank_states:
            if state.species_id in protein_ids:
                if state.contributes_to_signal:
                    protein_contributes = True

        # check if the target species is present
        target_species_present = False
        for state in photo_measurement.blank_states:
            if state.species_id == target_species:
                target_species_present = True

        if contains_protein and not protein_contributes and target_species_present:
            logger.debug(f"Well {well.id} is catalyzed.")
            return True

        return False

    @staticmethod
    def add_to_species_data(
        measurement: pe.Measurement,
        init_conditions: list[InitCondition],
        time_unit: UnitDefinition,
    ):
        for condition in init_conditions:
            measurement.add_to_species_data(
                species_id=condition.species_id,
                initial=condition.init_conc,
                prepared=condition.init_conc,
                data_unit=EnzML_UnitDef(**condition.conc_unit.model_dump()),
                time_unit=EnzML_UnitDef(**time_unit.model_dump()),
                data_type=DataTypes.CONCENTRATION,
            )

    @staticmethod
    def map_protein(protein: Protein, vessel_id: str) -> pe.Protein:
        if protein.ld_id_url:
            return pe.Protein(
                id=protein.id,
                ld_id=protein.ld_id_url,
                name=protein.name,
                constant=protein.constant,
                sequence=protein.sequence,
                vessel_id=vessel_id,
            )
        else:
            return pe.Protein(
                id=protein.id,
                name=protein.name,
                constant=protein.constant,
                sequence=protein.sequence,
                vessel_id=vessel_id,
            )

    @staticmethod
    def map_small_molecule(molecule: Molecule, vessel_id: str) -> pe.SmallMolecule:
        if molecule.ld_id_url:
            return pe.SmallMolecule(
                id=molecule.id,
                ld_id=molecule.ld_id_url,
                name=molecule.name,
                constant=molecule.constant,
                vessel_id=vessel_id,
            )
        else:
            return pe.SmallMolecule(
                vessel_id=vessel_id,
                id=molecule.id,
                name=molecule.name,
                constant=molecule.constant,
            )

    def _handle_wavelength(self):
        """
        Checks if a wavelength was specified and if not, sets it to the only wavelength measured.
        If multiple wavelengths were measured, an error is raised.
        """
        if isinstance(self.wavelength, float):
            return

        # check that all measurements in the wells have only one wavelength
        wavelengths = set()
        for well in self.plate.wells:
            for meas in well.measurements:
                wavelengths.add(meas.wavelength)

        if len(wavelengths) > 1:
            raise ValueError("Multiple wavelengths were measured. Please specify one.")

        self.wavelength = wavelengths.pop()

    def _init_calibrators(self):
        """Initializes calibrators for all molecules with a standard."""

        for molecule in self.molecules:
            assert molecule.id not in self.calibrator_dict, (
                f"Calibrator for molecule {molecule.id} already exists in calibrator_dict."
            )

            if not molecule.standard:
                continue

            calibrator = Calibrator.from_standard(molecule.standard)

            self.calibrator_dict[molecule.id] = calibrator

            logger.debug(f"Initialized calibrator for molecule {molecule.id}")
