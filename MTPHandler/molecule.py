from calipytion.model import Standard
from calipytion.model import UnitDefinition as CalUnit
from calipytion.tools.calibrator import Calibrator
from calipytion.units import C
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from MTPHandler.model import UnitDefinition


class Molecule(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assigment=True,
        use_enum_values=True,
    )  # type: ignore

    id: str = Field(
        description="ID of the molecule",
    )
    pubchem_cid: int = Field(
        description="PubChem CID of the molecule",
    )
    name: str = Field(
        description="Name of the molecule",
    )
    init_conc: float | None = Field(
        description="Initial concentration of the molecule at t=0", default=None
    )
    conc_unit: UnitDefinition | None = Field(
        description="Unit of the concentration", default=None
    )
    wavelength: float | None = Field(
        description="Wavelength at which the molecule was detected", default=None
    )
    standard: Standard | None = Field(
        description="Standard instance associated with the molecule", default=None
    )
    constant: bool = Field(
        description="Boolean indicating whether the molecule concentration is constant throughout the experiment",
        default=True,
    )

    # @model_validator(mode="before")
    # @classmethod
    # def get_molecule_name(cls, data: Any) -> Any:
    #     """Retrieves the molecule name from the PubChem database based on the PubChem CID."""

    #     if "name" not in data:
    #         data["molecule_name"] = pubchem_request_molecule_name(data["pubchem_cid"])
    #     return data

    # # validator that if a standard is provided, the retention time must be defined and vice versa

    # @model_validator(mode="before")
    # @classmethod
    # def validate_standard_and_retention_time(cls, data: Any) -> Any:
    #     if data.get("standard") and data.get("retention_time"):
    #         assert data["standard"].retention_time == data["retention_time"], """
    #         The retention time of the standard and the molecule must be the same.
    #         """

    @classmethod
    def from_standard(
        cls, standard: Standard, init_conc: float, conc_unit: UnitDefinition
    ):
        """Creates a Molecule instance from a Standard instance."""

        assert standard.retention_time, """
        The retention time of the standard needs to be defined. 
        Specify the `retention_time` attribute of the standard.
        """

        return cls(
            id=standard.molecule_id,
            pubchem_cid=standard.pubchem_cid,
            name=standard.molecule_name,
            init_conc=init_conc,
            conc_unit=conc_unit,
            standard=standard,
        )

    def create_standard(
        self,
        areas: list[float],
        concs: list[float],
        conc_unit: UnitDefinition,
        ph: float,
        temperature: float,
        temp_unit: CalUnit = C,
        visualize: bool = True,
    ) -> Standard:
        """Creates a linear standard from the molecule's calibration data."""

        calibrator = Calibrator(
            molecule_id=self.id,
            pubchem_cid=self.pubchem_cid,
            molecule_name=self.name,
            wavelength=self.wavelength,
            concentrations=concs,
            conc_unit=CalUnit(**conc_unit.model_dump()),
            signals=areas,
        )
        calibrator.models = []
        model = calibrator.add_model(
            name="linear",
            signal_law=f"{self.id} * a",
        )

        calibrator.fit_models()
        model.calibration_range.conc_lower = 0.0
        model.calibration_range.signal_lower = 0.0

        if visualize:
            calibrator.visualize()

        standard = calibrator.create_standard(
            model=model,
            ph=ph,
            temperature=temperature,
            temp_unit=CalUnit(**temp_unit.model_dump()),
        )

        # check if the `conc` attribute of the molecule is defined and if, it must have the same baseunit names as the calibration unit
        if self.conc_unit:
            try:
                for idx, unit in enumerate(self.conc_unit.base_units):
                    assert (
                        unit.kind == conc_unit.base_units[idx].kind
                        and unit.exponent == conc_unit.base_units[idx].exponent
                    ), """
                    Units dont match.
                    """
            except AssertionError:
                logger.warning(
                    f"The concentration unit of the molecule {self.name} does not match the calibration unit defined in its standard. Conc unit of the molecule was set to {conc_unit}."
                )
                self.conc_unit = conc_unit
                self.init_conc = None
        else:
            self.conc_unit = conc_unit

        self.standard = standard

        return standard


class Protein(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assigment=True,
        use_enum_values=True,
    )  # type: ignore

    id: str = Field(
        description="ID of the Protein",
    )
    name: str = Field(
        description="Name of the protein",
    )
    init_conc: float = Field(
        description="Initial concentration of the protein at t=0",
    )
    conc_unit: UnitDefinition = Field(
        description="Unit of the concentration",
    )
    sequence: str | None = Field(
        description="Amino acid sequence of the protein",
        default=None,
    )
    organism: str | None = Field(
        description="Organism from which the protein originates",
        default=None,
    )
    organism_tax_id: str | None = Field(
        description="Taxonomic ID of the organism",
        default=None,
    )
    constant: bool = Field(
        description="Boolean indicating whether the protein concentration is constant",
        default=True,
    )
