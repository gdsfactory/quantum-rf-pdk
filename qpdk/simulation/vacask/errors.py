"""Exceptions raised by the VACASK simulator wrappers."""


class VacaskError(RuntimeError):
    """Raised when VACASK is missing, a simulation fails, or a raw file is malformed."""
