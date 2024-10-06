import numpy as np
from typing import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class NumpyArrayField:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(np.ndarray),
                core_schema.chain_schema(
                    [
                        core_schema.list_schema(),
                        core_schema.no_info_plain_validator_function(np.array),
                    ]
                ),
            ]
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, (list, tuple)):
            return np.array(v)
        if isinstance(v, np.ndarray):
            return v
        raise ValueError("Invalid value for numpy array")
