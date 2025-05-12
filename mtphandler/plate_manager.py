from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any, Literal, Optional, Tuple, get_args

import numpy as np
import pandas as pd
from calipytion import Calibrator
from loguru import logger
from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, Field, model_validator
from pyenzyme import EnzymeMLDocument
from rich import print

from mtphandler.model import (
    BlankState,
    PhotometricMeasurement,
    Plate,
    Well,
)
from mtphandler.molecule import Molecule, Protein
from mtphandler.tools import (
    get_measurement,
    get_species_condition,
    handle_blank_status,
    measurement_is_blanked_for,
    pubchem_request_molecule_name,
    well_contains_species,
)
from mtphandler.visualize import visualize_plate

ASSIGN_CASE = Literal["rows", "columns", "all", "all except"]
ASSIGN_CASE_VALUES: Tuple[ASSIGN_CASE, ...] = get_args(ASSIGN_CASE)


class PlateManager(BaseModel):
    name: str = Field(
        ...,
        description="Name of the plate",
    )
    plate: Plate = Field(
        ...,
        description="Plate object",
    )
    molecules: list[Molecule] = Field(
        default=[],
        description="List of molecules",
    )
    proteins: list[Protein] = Field(
        default=[],
        description="List of proteins",
    )

    @model_validator(mode="before")
    @classmethod
    def give_name_to_plate(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "name" not in data or data["name"] is None:
                data["name"] = "MTP assay"
        return data

    @staticmethod
    def config_logger(
        log_path: Optional[str] = None,
        level: str = "INFO",
        to_stdout: bool = False,
    ):
        """
        Configures the internal Loguru logger.

        Parameters:
            log_path (str, optional): Path to the log file. If None, file logging is disabled.
            level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
            to_stdout (bool): If True, log messages will also be forwarded to stdout.

        Usage (in a Jupyter notebook or script):
            >>> from mtphandler import handler
            >>> handler.config_logger(log_path="app.log", level="DEBUG", to_stdout=True)
            >>> handler.logger.info("This message will go to the file and stdout.")
        """
        logger.remove()

        if to_stdout:
            logger.add(
                sys.stdout,
                level=level,
                colorize=True,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
            )

        if log_path:
            logger.add(
                log_path,
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                rotation="10 MB",
                retention="10 days",
            )

        logger.debug("Logger configured")

    def define_molecule(
        self,
        id: str,
        pubchem_cid: int,
        name: str | None = None,
        constant: bool = False,
    ) -> Molecule:
        """Defines a molecule which can be used to assign to wells on the plate.
        If no name is provided, the molecule name is retrieved from the PubChem database.
        If the molecule is not known in the PubChem database, please specify `pubchem_cid=-1`.

        Args:
            id (str): Internal identifier of the molecule such as `s0` or `ABTS`.
            pubchem_cid (int): PubChem CID of the molecule.
            name (str | None, optional): Name of the molecule. Defaults to None.
            constant (bool, optional): Indicates whether the molecule concentration is constant throughout the experiment. Defaults to False.

        Raises:
            ValueError: If the PubChem CID is not an integer.
            ValueError: If the name is not provided and the PubChem CID is not available.

        Returns:
            Molecule: Molecule object.
        """

        logger.debug(f"Defining molecule {id} with PubChem CID {pubchem_cid}")

        if not isinstance(pubchem_cid, int):
            raise ValueError("PubChem CID must be an integer.")

        if name is None:
            if pubchem_cid != -1:
                name = pubchem_request_molecule_name(pubchem_cid)
            else:
                raise ValueError(
                    "Name must be provided if PubChem CID is not available."
                )

        molecule = Molecule(
            id=id,
            pubchem_cid=pubchem_cid,
            name=name,
            constant=constant,
        )

        self._update_molecule(molecule)

        return molecule

    # Adders for species, molecules and proteins
    def add_molecule(
        self,
        molecule: Molecule,
        constant: bool | None = None,
    ) -> None:
        """Adds a molecule to the list of molecules. Allows to update the `constant` attribute of the molecule.

        Args:
            molecule (Molecule): Molecule object to add to the list of molecules.
            constant (bool | None, optional): Indicates whether the `constant` attribute of the molecule should be updated. Defaults to None.
        """
        if constant is not None:
            molecule = molecule.model_copy(update={"constant": constant})

        self._update_molecule(molecule)

    def _update_molecule(self, molecule) -> None:
        """Updates the molecule if it already exists in the list of molecules.
        Otherwise, the molecule is added to the list of species."""
        for idx, mol in enumerate(self.molecules):
            if mol.id == molecule.id:
                self.molecules[idx] = molecule
                assert self.molecules[idx] is molecule
                return

        self.molecules.append(molecule)

    def define_protein(
        self,
        id: str,
        name: str,
        uniprot_id: str | None = None,
        sequence: str | None = None,
        constant: bool = True,
    ) -> Protein:
        """Defines a protein which can be used to assign to wells on the plate.

        Args:
            id (str): Internal identifier of the protein such as `p0`, `MAT_K78M` or `GFP`.
            name (str): Name of the protein.
            uniprot_id (str | None, optional): UniProt ID of the protein. Defaults to None.
            sequence (str | None, optional): Amino acid sequence of the protein. Defaults to None.
            constant (bool, optional): Indicates whether the protein concentration is constant throughout the experiment. Defaults to True.

        Returns:
            Protein: Protein object.
        """
        protein = Protein(
            id=id,
            name=name,
            uniprot_id=uniprot_id,
            sequence=sequence,
            constant=constant,
        )

        self._update_protein(protein)

        return protein

    def add_protein(
        self,
        protein: Protein,
        constant: bool | None = None,
    ) -> None:
        """Adds a protein to the list of proteins. Allows to update the `constant` attribute of the protein.

        Args:
            protein (Protein): Protein object to add to the list of proteins.
            constant (bool | None, optional): Indicates whether the `constant` attribute of the protein should be updated. Defaults to None.
        """
        if constant is not None:
            protein = protein.model_copy(update={"constant": constant})

        self._update_protein(protein)

    def _update_protein(self, protein) -> None:
        """Updates the protein if it already exists in the list of proteins."""
        for idx, prot in enumerate(self.proteins):
            if prot.id == protein.id:
                self.proteins[idx] = protein
                assert self.proteins[idx] is protein
                return

        self.proteins.append(protein)

    # Assign species and conditions to wells
    def assign_init_conditions(
        self,
        species: Molecule | Protein,
        init_conc: float | list[float],
        conc_unit: UnitDefinitionAnnot,
        to: ASSIGN_CASE,
        ids: Optional[str | int | list[str] | list[int]] = None,
        contributes_to_signal: Optional[bool] = None,
        silent: bool = False,
    ):
        """
        Assigns a `Molecule` or `Protein` to specific wells on the plate based on the provided criteria.
        In this way the initial concentration of the species can be set for the respective wells in a row,
        column, all wells or all wells except for the specified. During the assignment, either an array of
        initial concentrations or a single initial concentration can be provided. If a single initial
        concentration is provided, it is assigned to all wells of e.g., a row or column.
        If an array of initial concentrations is provided, the length of the array must match the number of
        wells in the row or column.

        Tip:
            For complex assignment scenarios, consider using the `assign_init_conditions_from_spreadsheet` function.

        Args:
            species (Molecule | Protein): The species to assign to the wells.
            init_conc (float | list[float]): The initial concentration(s) of the species.
            conc_unit (UnitDefinitionAnnot): The unit of concentration.
            to (ASSIGN_CASE): The target location(s) for assigning the species. It should be one of the allowed cases.
            ids (str | int | list[str] | list[int], optional): The ID(s) of the target wells, rows, or columns. Defaults to None.
            contributes_to_signal (bool, optional): Indicates if the assigned species contributes to the signal.
                Defaults to None.
            silent (bool, optional): If True, no output is printed. Defaults to False.

        Raises:
            AttributeError: If the species does not exist in the list of molecules or proteins.
            AttributeError: If the 'to' argument is not a valid `ASSIGN_CASE`.

        Returns:
            None
        """

        # Handle species
        if isinstance(species, str):
            species = self.get_species(species)
        elif isinstance(species, (Molecule, Protein)):
            pass
        else:
            raise AttributeError(
                """Argument 'species' must reference an `id` of a molecule or protein from the list of molecules or proteins of the `MTPHandler`."""
            )

        if to not in ASSIGN_CASE_VALUES:
            raise AttributeError(f"Argument 'to' must be one of {ASSIGN_CASE_VALUES}.")

        if not isinstance(init_conc, list):
            init_conc = [init_conc]

        if not isinstance(ids, list) and isinstance(ids, (int, str)):
            ids = [ids]  # type: ignore

        if to == "all":
            if isinstance(init_conc, list):
                if len(init_conc) == 1:
                    init_conc = init_conc[0]
            assert isinstance(init_conc, (float, int)), (
                "Argument 'init_conc' must be a float or an integer."
            )

            self._assign_to_all(
                species=species,
                init_conc=float(init_conc),
                conc_unit=conc_unit,
                contributes_to_signal=contributes_to_signal,
                silent=silent,
            )

        elif to == "columns":
            assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), (
                "Argument 'ids' must be a list of integers when 'to' is set to 'columns'."
            )

            self._assign_to_columns(
                column_ids=ids,  # type: ignore
                species=species,
                init_concs=init_conc,
                conc_unit=conc_unit,
                contributes_to_signal=contributes_to_signal,
                silent=silent,
            )

        elif to == "rows":
            assert isinstance(ids, list) and all(isinstance(i, str) for i in ids), (
                "Argument 'ids' must be a list of strings when 'to' is set to 'rows'."
            )

            self._assign_species_to_rows(
                row_ids=ids,  # type: ignore
                species=species,
                init_concs=init_conc,
                conc_unit=conc_unit,
                contributes_to_signal=contributes_to_signal,
                silent=silent,
            )

        else:
            if isinstance(init_conc, list):
                if len(init_conc) == 1:
                    init_conc = init_conc[0]

            assert isinstance(init_conc, float), (
                "Argument 'init_conc' must be a float when 'to' is set to 'all_except'."
            )

            self._assign_species_to_all_except(
                well_ids=ids,  # type: ignore
                species=species,
                init_conc=init_conc,
                conc_unit=conc_unit,
                contributes_to_signal=contributes_to_signal,
                silent=silent,
            )

    def _assign_to_all(
        self,
        species: Molecule | Protein,
        init_conc: float,
        conc_unit: UnitDefinitionAnnot,
        contributes_to_signal: bool | None,
        silent: bool,
    ):
        for well in self.plate.wells:
            well.add_to_init_conditions(
                species_id=species.id,
                init_conc=init_conc,
                conc_unit=conc_unit,
            )

            handle_blank_status(well, species.id, init_conc, contributes_to_signal)

        if not silent:
            print(
                f"Assigned [bold magenta]{species.name}[/] ({species.id}) with"
                f" {init_conc} {conc_unit} to all wells."
            )

    def get_calibrator(
        self,
        molecule: Molecule,
        cutoff: float | None = None,
        wavelength: float | None = None,
    ) -> Calibrator:
        """Initialize a CaliPytion `Calibrator` for a molecule on the plate.
        The calibrator allows eighter to proceed with predefined suitable calibration models or
        to define and fit custom models. For more information on the CaliPytion, please refer to the
        (documentation)[https://fairchemistry.github.io/CaliPytion/usage/]

        Args:
            molecule (Molecule): The molecule for which to initialize the calibrator.
            cutoff (float | None, optional): The cutoff value for the calibration. Absorption values
                above the cutoff are not considered for the calibration. Defaults to None.
            wavelength (float | None, optional): The wavelength at which to initialize the calibrator.
                If only one wavelength was measured, the wavelength is automatically set. Defaults to None.

        Returns:
            Calibrator: Calibrator object.
        """
        from mtphandler.ioutils.calipytion import initialize_calibrator

        if wavelength is None:
            wavelength = self._handle_wavelength()

        return initialize_calibrator(
            plate=self.plate,
            wavelength=wavelength,
            molecule=molecule,
            protein_ids=[protein.id for protein in self.proteins],
            cutoff=cutoff,
        )

    def _assign_to_columns(
        self,
        column_ids: list[int],
        species: Molecule | Protein,
        init_concs: list[float],
        conc_unit: UnitDefinitionAnnot,
        contributes_to_signal: bool | None,
        silent: bool,
    ):
        # Handle column_ids
        if not all([isinstance(column_id, int) for column_id in column_ids]):
            raise AttributeError("Argument 'column_ids' must be a list of integers.")

        columns = []
        for column_id in column_ids:
            wells = [well for well in self.plate.wells if well.x_pos + 1 == column_id]
            wells = sorted(wells, key=lambda x: x.y_pos)
            columns.append(wells)

        # assert thal all columns are the same size
        assert all([len(column) == len(columns[0]) for column in columns]), (
            "All columns must be the same size. "
        )

        # Handle init_concs
        if len(init_concs) == 1:
            init_concs = init_concs * len(columns[0])

        for wells in columns:
            assert len(init_concs) == len(wells), f"""
            Number of initial concentrations ({len(init_concs)}) does not match number
            of wells ({len(wells)}) in columns ({column_ids}).
            """

            for well, init_conc in zip(wells, init_concs):
                well.add_to_init_conditions(
                    species_id=species.id,
                    init_conc=init_conc,
                    conc_unit=conc_unit,
                )

                handle_blank_status(well, species.id, init_conc, contributes_to_signal)

        if not silent:
            print(
                f"Assigned [bold magenta]{species.name}[/] ({species.id}) with"
                f" concentrations of {init_concs} {conc_unit} to columns {column_ids}."
            )

    def _assign_species_to_rows(
        self,
        row_ids: list[str],
        species: Molecule | Protein,
        init_concs: list[float],
        conc_unit: UnitDefinitionAnnot,
        contributes_to_signal: bool | None,
        silent: bool,
    ):
        # Handle row_ids
        if isinstance(row_ids, str):
            row_ids = [row_ids]

        if not all([isinstance(row_id, str) for row_id in row_ids]):
            raise AttributeError("Argument 'row_ids' must be a list of strings.")

        rows = []
        for row_id in row_ids:
            wells = [well for well in self.plate.wells if row_id in well.id]
            wells = sorted(wells, key=lambda x: x.x_pos)
            rows.append(wells)

        for wells in rows:
            assert len(init_concs) == len(wells), f"""
            Number of initial concentrations ({len(init_concs)}) does not match number
            of wells ({len(wells)}) in rows ({row_ids}).
            """

            for well, init_conc in zip(wells, init_concs):
                well.add_to_init_conditions(
                    species_id=species.id,
                    init_conc=init_conc,
                    conc_unit=conc_unit,
                )

                handle_blank_status(well, species.id, init_conc, contributes_to_signal)

        if not silent:
            print(
                f"Assigned [bold magenta]{species.name}[/] ({species.id}) with"
                f" {init_concs} {conc_unit} to rows {row_ids}."
            )

    def _assign_species_to_all_except(
        self,
        well_ids: list[str],
        species: Molecule | Protein,
        init_conc: float,
        conc_unit: UnitDefinitionAnnot,
        contributes_to_signal: bool | None,
        silent: bool,
    ):
        # validate all well_id exist
        for well_id in well_ids:
            if not self._well_id_exists(well_id):
                raise AttributeError(f"Well ID '{well_id}' not found on the plate.")

        wells = (well for well in self.plate.wells if well.id not in well_ids)
        for well in wells:
            well.add_to_init_conditions(
                species_id=species.id,
                init_conc=init_conc,
                conc_unit=conc_unit,
            )

            handle_blank_status(well, species.id, init_conc, contributes_to_signal)

        if not silent:
            print(
                f"Assigned [bold magenta]{species.name}[/] ({species.id}) with"
                f" {init_conc} {conc_unit} to all wells except {well_ids}."
            )

    def create_assignment_spreadsheet(
        self,
        path: str = "assignment.xlsx",
        overwrite: bool = False,
    ):
        """Create an Excel spreadsheet for assigning initial concentrations. The spreadsheet
        contains a separate sheet for each species defined on the plate, with validation
        to allow only numerical values in the input cells and prevent changes to all other cells.

        Args:
            path (str, optional): Path to save the assignment spreadsheet. Defaults to "assignment.xlsx".
            overwrite (bool, optional): If True, the file is overwritten if it already exists. Defaults to False.
        """

        if not overwrite:
            if os.path.exists(path):
                print(f"File {path} already exists. Set 'overwrite=True' to overwrite.")
                return
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df = pd.DataFrame(
                index=[chr(65 + i) for i in range(8)],
                columns=[str(i + 1) for i in range(12)],
            )
            df.to_excel(writer, sheet_name="pH", index=True)
            worksheet = writer.sheets["pH"]

            worksheet.data_validation(
                "B2:M9",
                {
                    "validate": "decimal",
                    "criteria": "between",
                    "minimum": 0,
                    "maximum": 12,
                    "error_message": "pH must be a numerical value between 0 and 12",
                },
            )
            for species in self.molecules + self.proteins:
                df = pd.DataFrame(
                    index=[chr(65 + i) for i in range(8)],
                    columns=[str(i + 1) for i in range(12)],
                )

                df.to_excel(writer, sheet_name=species.id, index=True)

                worksheet = writer.sheets[species.id]

                worksheet.data_validation(
                    "B2:M9",
                    {
                        "validate": "decimal",
                        "criteria": "between",
                        "minimum": 0,
                        "maximum": 1e18,
                        "error_message": "This cell only accepts numerical values",
                    },
                )

        print(f"Created assignment spreadsheet at {path}.")

    def assign_init_conditions_from_spreadsheet(
        self,
        conc_unit: UnitDefinitionAnnot,
        path: str,
        header: int = 0,
        index: int = 0,
        silent: bool = False,
    ):
        """Assign initial concentrations from an Excel spreadsheet to the wells on the plate.

        Note:
            This function goes through the sheets in an excel spreadsheet. If the sheet name
            matches the id of a protein or molecule defined for the plate, the initial concentration
            form the plate map in the excel spreadsheet is assigned to the respective well.

            The excel spreadsheet must have the following structure:

            - The first row must contain the column numbers from 1 to 12.
            - The first column must contain the row letters from A to H.
            - If a cell is left empty for a species, the species is not assigned to the well.
            - If the initial concentration is `0`, the species is added to the well. This is useful for
                specifying a product which is not present in the initial reaction mixture, but is formed
                during the reaction.

        Args:
            conc_unit (UnitDefinitionAnnot): The unit of concentration.
            path (str): Path to the Excel spreadsheet.
            header (int, optional): Row to use as the column names. Defaults to 0.
            index (int, optional): Column to use as the row labels. Defaults to 0.
            silent (bool, optional): If True, no output is printed. Defaults to False.
        """
        # get excel sheet names
        count = 0
        sheet_names = pd.ExcelFile(path).sheet_names

        species_matches: set[str] = set()
        for protein in self.proteins:
            if protein.id.lower() in [sheet.lower() for sheet in sheet_names]:
                species_matches.add(protein.id)
        for molecule in self.molecules:
            if molecule.id.lower() in [sheet.lower() for sheet in sheet_names]:
                species_matches.add(molecule.id)

        for species_id in species_matches:
            df = pd.read_excel(
                io=path, header=header, index_col=index, sheet_name=species_id
            )
            for well in self.plate.wells:
                init_conc = df.iloc[well.y_pos, well.x_pos]

                if np.isnan(init_conc):
                    continue

                well.add_to_init_conditions(
                    species_id=species_id,
                    init_conc=init_conc,
                    conc_unit=conc_unit,
                )
                count += 1

                handle_blank_status(
                    well, species_id, init_conc, contributes_to_signal=None
                )

        ph_set = False
        if "pH" in sheet_names:
            df = pd.read_excel(io=path, header=header, index_col=index, sheet_name="pH")
            for well in self.plate.wells:
                ph = df.iloc[well.y_pos, well.x_pos]

                if np.isnan(ph):
                    continue

                well.ph = ph

                count += 1
                ph_set = True

        if not silent:
            ph_message = "[bold magenta]pH[/] and" if ph_set else ""
            print(
                f"📍 Assigned {count} initial concentration coditions for {ph_message} [bold magenta]{list(species_matches)}[/]"
                f" from {path} to the plate."
            )

    def set_absorption_contribution(
        self,
        species: Molecule | Protein,
        contributes_to_signal: bool,
        wavelength: float | None = None,
        silent: bool = False,
    ):
        """Set the contribution of a species to the signal in all wells.

        Args:
            species (Molecule | Protein): The species for which to set the contribution to the signal.
            contributes_to_signal (bool): If True, the species contributes to the signal. If False, the species does not contribute to the signal.
            wavelength (float | None, optional): The wavelength at which to set the contribution to the signal. Defaults to None.
            silent (bool, optional): If True, no output is printed. Defaults to False.
        """
        if wavelength is None:
            try:
                wavelength = self._handle_wavelength()
            except ValueError:
                raise ValueError(
                    "Multiple wavelengths were measured. Please specify one."
                )

        for well in self.plate.wells:
            if not well_contains_species(well, species.id):
                continue

            for measurement in well.measurements:
                if measurement.wavelength != wavelength:
                    continue

                for state in measurement.blank_states:
                    if state.species_id == species.id:
                        state.contributes_to_signal = contributes_to_signal

        if not silent:
            print(
                f"Set signal contribution of [bold magenta]{species.name}[/] ({species.id}) at"
                f" {wavelength} nm to {contributes_to_signal}."
            )

    def get_well(self, id: str) -> Well:
        """Get a well from the plate by its id.

        Args:
            id (str): The id of the well.

        Raises:
            ValueError: If the well with the given id is not found.

        Returns:
            Well: The well object.
        """

        for well in self.plate.wells:
            if well.id.lower() == id.lower():
                return well

        raise ValueError(f"Well {id} not found")

    def get_species(self, id: str) -> Protein | Molecule:
        """Get a species from the list of molecules and proteins by its id.

        Args:
            id (str): The id of the species.

        Raises:
            ValueError: If the species with the given id is not found.

        Returns:
            Protein | Molecule: The species object.
        """
        for protein in self.proteins:
            if protein.id == id:
                return protein
        for molecule in self.molecules:
            if molecule.id == id:
                return molecule

        raise ValueError(f"Species {id} not found")

    def visualize(
        self,
        zoom: bool = False,
        wavelengths: list[float] = [],
        darkmode: bool = False,
    ):
        """Visualize the plate.

        Args:
            zoom (bool, optional): If False, the scaling of the signal (y-axis) is the same for all wells.
                If True, the scaling is adjusted for each well. Defaults to False.
            wavelengths (list[float], optional): Only visualize the signal at the specified wavelengths.
                If not specified, all wavelengths are visualized. Defaults to [].
            darkmode (bool, optional): If True, the plot is displayed in dark mode. Defaults to False.
        """

        visualize_plate(
            self.plate,
            zoom=zoom,
            wavelengths=wavelengths,
            darkmode=darkmode,
            name=self.name,
        )

    def _handle_wavelength(self) -> float:
        """
        If only one wavelength was measured, the wavelength is returned.
        If multiple wavelengths were measured, an error is raised.
        """

        # check that all measurements in the wells have only one wavelength
        wavelengths = set()
        for well in self.plate.wells:
            for meas in well.measurements:
                wavelengths.add(meas.wavelength)

        if len(wavelengths) > 1:
            raise ValueError("Multiple wavelengths were measured. Please specify one.")

        return wavelengths.pop()

    def _find_blanking_wells(
        self,
        target: Molecule | Protein,
        wavelength: float,
    ) -> list[Well]:
        wells = []

        if wavelength is None:
            wavelength = self._handle_wavelength()

        protein_ids = [protein.id for protein in self.proteins]
        molecules_ids = [molecule.id for molecule in self.molecules]

        # find wells that contain the target species with a concentration above zero
        for well in self.plate.wells:
            if not well_contains_species(well, target.id, conc_above_zero=True):
                continue

            # Molecule controls can not include proteins
            if target.id in molecules_ids and any(
                [
                    well_contains_species(well, protein_id, conc_above_zero=True)
                    for protein_id in protein_ids
                ]
            ):
                continue

            for measurement in well.measurements:
                if measurement.wavelength != wavelength:
                    continue

                # sanity check, species should be present in blank states
                assert target.id in [
                    state.species_id for state in measurement.blank_states
                ], f"Species {target.id} not found in well {well.id}."

                # check is species contributes to signal (== is already blanked)
                if measurement_is_blanked_for(measurement, target.id):
                    wells.append(well)

        return wells

    def slice_data(
        self,
        start: float,
        end: float,
    ):
        """Slices the time and absorption data of all wells in the plate
        that only contains the data between the start and end time.

        Args:
            start (float): Start time of the slice.
            end (float): End time of the slice.
        """

        for well in self.plate.wells:
            for meas in well.measurements:
                # find the index of the start and end time
                start_idx = np.where(np.array(meas.time) >= start)[0][0]
                end_idx = np.where(np.array(meas.time) <= end)[0][-1]

                # slice the time and absorption data
                meas.time = meas.time[start_idx:end_idx]
                meas.absorption = meas.absorption[start_idx:end_idx]

    def blank_species(
        self,
        species: Molecule | Protein,
        wavelength: float | None = None,
        control_wells: list[str] = [],
        silent: bool = False,
    ):
        """Blank the signal contribution of a species at a given wavelength.
        Therefore, control wells of that species must be present on the plate.

        Args:
            species (Molecule | Protein): The species to blank.
            wavelength (float): The wavelength at which to blank the species.
            control_wells (list[str] | None, optional): List of well ids that are used to calculate the absorption contribution of the species.
                Defaults to None.
            silent (bool, optional): If True, no output is printed. Defaults to False.

        Raises:
            ValueError: If no wells are found to calculate the absorption contribution of the species.
        """

        if wavelength is None:
            wavelength = self._handle_wavelength()

        if not control_wells:
            blanking_wells = self._find_blanking_wells(
                target=species, wavelength=wavelength
            )
        else:
            blanking_wells = [self.get_well(well_id) for well_id in control_wells]

        if not blanking_wells:
            logger.warning(
                f"No wells found to calculate the absorption contribution of the species {species.name} ({species.id}). Was the species already blanked?"
            )
            return

        # get mapping of concentration to blank wells
        conc_blank_mapping = self._get_conc_blank_mapping(
            wells=blanking_wells, species=species, wavelength=wavelength
        )

        self._apply_blank(
            species=species,
            conc_blank_mapping=conc_blank_mapping,
            wavelength=wavelength,
        )

    def _apply_blank(
        self,
        species: Molecule | Protein,
        conc_blank_mapping: dict[float, float],
        wavelength: float,
    ):
        """Apply the blanking to the absorption data of a well.

        Args:
            species (Molecule | Protein): The species to blank.
            conc_blank_mapping (dict[float, float]): Mapping of init concentration of the species to mean absorption.
            wavelength (float): The wavelength at which to blank the species.
        """
        well_blanked_count = 0

        for well_id, well in enumerate(self.plate.wells):
            for meas_id, measurement in enumerate(well.measurements):
                if measurement.wavelength != wavelength:
                    continue

                try:
                    init_condition = get_species_condition(well, species.id)
                except ValueError:
                    continue

                for state_id, blank_state in enumerate(measurement.blank_states):
                    if blank_state.species_id != species.id:
                        continue

                    if blank_state.contributes_to_signal:
                        self.plate.wells[well_id].measurements[meas_id].absorption = [
                            absorption - conc_blank_mapping[init_condition.init_conc]
                            for absorption in measurement.absorption
                        ]

                        self.plate.wells[well_id].measurements[meas_id].blank_states[
                            state_id
                        ].contributes_to_signal = False

                        well_blanked_count += 1

        logger.info(f"Blanked {well_blanked_count} wells containing {species.name}.")

    def to_enzymeml(
        self,
        detected_molecule: Molecule,
        well_ids: list[str] | None = None,
        catalyzed_only: bool = True,
        name: str | None = None,
        to_concentration: bool = False,
        extrapolate: bool = False,
        wavelength: float | None = None,
        silent: bool = False,
    ) -> EnzymeMLDocument:
        """Convert the plate to an EnzymeML document.


        Args:
            name (str | None, optional): Name of the EnzymeML document. Defaults to the name of the plate.
            detected_molecule (Molecule): The molecule that was detected in the wells.
            well_ids (list[str] | None, optional): List of well ids to include in the EnzymeML document.
                If not provided, all wells are included. Defaults to None.
            to_concentration (bool, optional): If True, the signal is converted to concentration. Therefore,
                a calibrator must be defined for the respective molecule. Defaults to False.
            extrapolate (bool, optional): If True, and `to_concentration` is True, measured absorption values
                that are outside the range of the calibrator are extrapolated. Defaults to False.
            catalyzed_only (bool, optional): If True, only wells that contain the detected molecule and a
                protein are included in the EnzymeML document. Defaults to True.
            wavelength (float | None, optional): If multiple wavelengths were measured, the wavelength for
                which to convert the signal to concentration needs to be specified. Defaults to None.
            silent (bool, optional): If True, no output is printed. Defaults to False.

        Returns:
            EnzymeMLDocument: [`pyenzyme`](https://github.com/EnzymeML/PyEnzyme) `EnzymeMLDocument` object.
        """
        from mtphandler.ioutils.pyenzyme import Plate_to_EnzymeMLDocument

        if name is None:
            name = self.name

        converter = Plate_to_EnzymeMLDocument(
            name=name,
            plate=self.plate,
            well_ids=well_ids,
            molecules=self.molecules,
            detected_molecule=detected_molecule,
            proteins=self.proteins,
            to_concentration=to_concentration,
            extrapolate=extrapolate,
            catalyzed_only=catalyzed_only,
            wavelength=wavelength,
            silent=silent,
        )

        return converter.convert()

    def _well_id_exists(self, well_id: str) -> bool:
        """Check if a well with the given id exists in the plate."""
        return any([well_id in well.id for well in self.plate.wells])

    def _get_conc_blank_mapping(
        self,
        wells: list[Well],
        species: Protein | Molecule,
        wavelength: float,
    ) -> dict[float, float]:
        """Calculate the mean absorption of a species at different concentrations.

        Args:
            wells (list[Well]): List of wells to calculate the mean absorption for.
            species (Protein | Molecule): The species for which to calculate the mean absorption.
            wavelength (float): The wavelength at which to calculate the mean absorption.

        Returns:
            dict[float, float]: Mapping of concentration to mean absorption.
        """
        conc_to_absorptions = defaultdict(list)

        # Collect all absorption data per concentration
        for well in wells:
            condition = get_species_condition(well, species.id)
            absorption = get_measurement(well, wavelength).absorption
            conc_to_absorptions[condition.init_conc].append(absorption)

        conc_mean_blank_mapping = {}

        # Calculate mean absorption and standard deviation
        for conc, absorptions in conc_to_absorptions.items():
            mean_absorption = np.nanmean(absorptions)
            std_absorption = np.nanstd(absorptions)

            # Handle case where mean_absorption is zero to avoid division by zero
            if mean_absorption != 0:
                std_perc = float(abs(std_absorption / mean_absorption) * 100)
            else:
                std_perc = 0.0

            logger.info(
                f"Mean absorption of {species.name} ({species.id}) at"
                f" {conc} {condition.conc_unit.name}: {mean_absorption:.4f} ±"
                f" {std_perc:.0f}%  calculated based on wells"
                f" {[well.id for well in wells]}."
            )

            conc_mean_blank_mapping[conc] = mean_absorption.tolist()

        return conc_mean_blank_mapping

    @staticmethod
    def _species_contibutes(
        measurement: PhotometricMeasurement, species_id: str
    ) -> bool:
        species_contributes = [
            state.contributes_to_signal
            for state in measurement.blank_states
            if state.species_id == species_id
        ][0]

        return species_contributes

    @staticmethod
    def _get_blank_state(
        measurement: PhotometricMeasurement, species_id: str
    ) -> BlankState:
        for state in measurement.blank_states:
            if state.species_id == species_id:
                return state

        raise ValueError(f"Species {species_id} is not present in this well.")

    @classmethod
    def read_spectra_max_190(
        cls,
        path: str,
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.txt` file exported from a SpectraMax 190 software and create a PlateManager object.

        Args:
            path (str): Path to the SpectraMax 190 file.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.
            name (str | None, optional): Name of the plate. Defaults to None.

        Returns:
            PlateManager: PlateManager object.
        """
        from mtphandler.readers import read_spectra_max_190 as reader

        data: dict[str, Any] = {"plate": reader(path, ph)}

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_multiskan_spectrum_1500(
        cls,
        path: str,
        time: list[float],
        time_unit: UnitDefinitionAnnot,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "C",
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.txt` file exported from a Multiskan Spectrum 1500 and create a PlateManager object.

        Args:
            name (str): Name of the plate.
            path (str): Path to the Multiskan Spectrum 1500 file.
            time (list[float]): List of time points.
            time_unit (UnitDefinitionAnnot): Unit of time.
            temperature (float): Temperature of the measurements.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of temperature. Defaults to C.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.

        Returns:
            _type_: _description_
        """
        from mtphandler.readers import read_multiskan_spectrum_1500 as reader

        data: dict[str, Any] = {
            "plate": reader(
                path=path,
                time=time,
                time_unit=time_unit,
                temperature=temperature,
                temperature_unit=temperature_unit,
                ph=ph,
            )
        }

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_tecan_spark(
        cls,
        path: str,
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.xlsx` TECAN Spark file and create a PlateManager object.

        Args:
            path (str): Path to the TECAN Spark file.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.
            name (str | None, optional): Name of the plate. Defaults to None.

        Returns:
            PlateManager: PlateManager object.
        """
        from mtphandler.readers import read_tekan_spark as reader

        data: dict[str, Any] = {"plate": reader(path, ph)}

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_biotek(
        cls,
        path: str,
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.xlsx` file exported from a BioTek Epoch 2 software and create a PlateManager object.

        Args:
            path (str): Path to the BioTek Epoch 2 file.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.
            name (str | None, optional): Name of the plate. Defaults to None.

        Returns:
            PlateManager: PlateManager object.
        """
        from mtphandler.readers import read_biotek as reader

        data: dict[str, Any] = {"plate": reader(path, ph)}

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_tekan_magellan(
        cls,
        path: str,
        wavelength: float,
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.xlsx` file exported from a TECAN Magellan software and create a PlateManager object.

        Args:
            path (str): Path to the Magellan file.
            wavelength (float): The wavelength of the measurements.
            ph (Optional[float], optional): The pH value of the measurements. Defaults to None.
            name (Optional[str], optional): Name of the plate. Defaults to None.

        Returns:
            PlateManager: PlateManager object.
        """
        from mtphandler.readers import read_tekan_magellan as reader

        data: dict[str, Any] = {"plate": reader(path, wavelength, ph)}

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_new_device(
        cls,
        path: str,
        temperature: float,
        ph: float | None = None,
        name: str | None = None,
        temperature_unit: UnitDefinitionAnnot = "C",
    ) -> PlateManager:
        """Read a `*.xlsx` file exported from a new device and create a PlateManager object.

        Args:
            path (str): Path to the file.
            temperature (float): The temperature of the measurements.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.
            name (str | None, optional): Name of the plate. Defaults to None.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of temperature. Defaults to C.

        Returns:
            PlateManager: PlateManager object.
        """
        from mtphandler.readers import read_new_device as reader

        data: dict[str, Any] = {
            "plate": reader(
                path=path,
                temperature=[temperature],
                temperature_unit=temperature_unit,
                ph=ph,
            )
        }

        if name is not None:
            data["name"] = name

        return cls(**data)

    @classmethod
    def read_multiskan_sky(
        cls,
        path: str,
        ph: float | None = None,
        name: str | None = None,
    ) -> PlateManager:
        """Read a `*.xlsx` file exported from a Multiskan Sky and create a PlateManager object.

        Args:
            path (str): Path to the Multiskan Sky file.
            ph (float | None, optional): The pH value of the measurements. Defaults to None.
            name (str | None, optional): Name of the plate. Defaults to None.

        Returns:
            PlateManager: The PlateManager object.
        """
        from mtphandler.readers import read_multiskan_sky as reader

        data: dict[str, Any] = {"plate": reader(path, ph)}

        if name is not None:
            data["name"] = name

        return cls(**data)
