import re

from calipytion.model import Calibration
from calipytion.tools.calibrator import Calibrator
from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, ConfigDict, Field


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
    standard: Calibration | None = Field(
        description="Standard instance associated with the molecule", default=None
    )
    constant: bool = Field(
        description="Boolean indicating whether the molecule concentration is constant throughout the experiment",
        default=False,
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

    @property
    def ld_id_url(self) -> str | None:
        """Returns the URL of the PubChem page of the molecule based on the PubChem CID

        Returns:
            str | None: URL of the PubChem page of the molecule if the PubChem CID is defined, None otherwise.
        """

        if self.pubchem_cid == -1:
            return None

        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.pubchem_cid}"

    @classmethod
    def from_standard(
        cls, standard: Calibration, init_conc: float, conc_unit: UnitDefinitionAnnot
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
            standard=standard,
        )

    def create_standard(
        self,
        areas: list[float],
        concs: list[float],
        conc_unit: UnitDefinitionAnnot,
        ph: float,
        temperature: float,
        temp_unit: UnitDefinitionAnnot = "C",
        visualize: bool = True,
    ) -> Calibration:
        """Creates a linear standard from the molecule's calibration data."""

        calibrator = Calibrator(
            molecule_id=self.id,
            pubchem_cid=self.pubchem_cid,
            molecule_name=self.name,
            concentrations=concs,
            conc_unit=conc_unit.name,
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
            temp_unit=temp_unit.name,
        )

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
    uniprot_id: str | None = Field(
        description="UniProt ID of the protein",
        default=None,
    )
    name: str = Field(
        description="Name of the protein",
    )
    sequence: str | None = Field(
        description="Amino acid sequence of the protein",
        default=None,
    )
    constant: bool = Field(
        description="Boolean indicating whether the protein concentration is constant",
        default=True,
    )

    @property
    def ld_id_url(self) -> str | None:
        """Returns the URL of the UniProt page of the protein based on the protein ID

        Returns:
            str | None: URL of the UniProt page of the protein if the protein ID is defined, None otherwise.
        """

        uniprot_pattern = (
            r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
        )

        if re.fullmatch(uniprot_pattern, self.id) is None:
            return None
        else:
            return f"https://www.uniprot.org/uniprotkb/{self.id}/entry"
