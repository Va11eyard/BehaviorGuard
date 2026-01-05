"""Input validation and parsing for BehaviorGuard system."""

import json
from datetime import datetime
from typing import Union

from pydantic import ValidationError

from behaviorguard.models import (
    ErrorDetail,
    ErrorResponse,
    EvaluationInput,
    ValidationResult,
)


class InputValidator:
    """Validates and parses incoming JSON input."""

    def validate(self, raw_input: str) -> ValidationResult:
        """
        Validate JSON input against schema.

        Args:
            raw_input: Raw JSON string input

        Returns:
            ValidationResult with validation status and errors
        """
        errors = []

        # Check if input is valid JSON
        try:
            data = json.loads(raw_input)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False, errors=[f"Invalid JSON: {str(e)}"]
            )

        # Validate against Pydantic model
        try:
            EvaluationInput(**data)
            return ValidationResult(is_valid=True, errors=[])
        except ValidationError as e:
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                message = error["msg"]
                errors.append(f"{field}: {message}")

            return ValidationResult(is_valid=False, errors=errors)

    def parse(self, raw_input: str) -> Union[EvaluationInput, ErrorResponse]:
        """
        Parse and validate JSON input.

        Args:
            raw_input: Raw JSON string input

        Returns:
            EvaluationInput if valid, ErrorResponse if invalid
        """
        validation_result = self.validate(raw_input)

        if not validation_result.is_valid:
            return ErrorResponse(
                error=ErrorDetail(
                    type="ValidationError",
                    message="Input validation failed",
                    details=validation_result.errors,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
            )

        try:
            data = json.loads(raw_input)
            return EvaluationInput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            return ErrorResponse(
                error=ErrorDetail(
                    type="ValidationError",
                    message="Failed to parse input",
                    details=[str(e)],
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
            )
