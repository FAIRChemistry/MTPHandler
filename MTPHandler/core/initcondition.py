from typing import Dict, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.datatypes import Unit
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict


class InitCondition(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Description of the initial condition of a species in a well."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    species_id: str = element(
        description="Reference to species",
        tag="species_id",
        json_schema_extra=dict(),
    )

    init_conc: float = element(
        description="Initial concentration of the species",
        tag="init_conc",
        json_schema_extra=dict(),
    )

    conc_unit: Unit = element(
        description="Concentration unit",
        tag="conc_unit",
        json_schema_extra=dict(),
    )

    _repo: Optional[str] = PrivateAttr(
        default="https://github.com/FAIRChemistry/MTPHandler"
    )
    _commit: Optional[str] = PrivateAttr(
        default="b67724f080afb13c3b78cd2a559646f8b3f2e6e7"
    )

    _raw_xml_data: Dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _parse_raw_xml_data(self):
        for attr, value in self:
            if isinstance(value, (ListPlus, list)) and all(
                isinstance(i, _Element) for i in value
            ):
                self._raw_xml_data[attr] = [elem2dict(i) for i in value]
            elif isinstance(value, _Element):
                self._raw_xml_data[attr] = elem2dict(value)

        return self
