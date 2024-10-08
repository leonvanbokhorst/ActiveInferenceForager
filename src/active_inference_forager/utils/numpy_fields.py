import numpy as np
from typing import Any, Generator, Union
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class NumpyArrayField:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Get the core schema for Pydantic validation.

        Args:
            source_type (Any): The source type to validate.
            handler (GetCoreSchemaHandler): The handler for core schema.

        Returns:
            core_schema.CoreSchema: The core schema for validation.
        """
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
    def __get_validators__(cls) -> Generator:
        """
        Yield the validators for the field.

        Yields:
            Generator: A generator yielding validator functions.
        """
        yield cls.validate

    @classmethod
    def validate(cls, v: Union[list, tuple, np.ndarray]) -> np.ndarray:
        """
        Validate and convert the input to a numpy array.

        Args:
            v (Union[list, tuple, np.ndarray]): The input value to validate.

        Returns:
            np.ndarray: The validated numpy array.

        Raises:
            ValueError: If the input value is not valid.
        """
        if isinstance(v, (list, tuple)):
            return np.array(v)
        if isinstance(v, np.ndarray):
            return v
        raise ValueError(f"Invalid value for numpy array: {v} (type: {type(v)})")
