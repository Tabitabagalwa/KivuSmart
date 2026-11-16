class NasaApiError(RuntimeError):
    """A NASA service returned an error or could not be reached."""

    def __init__(self, service: str, message: str, status_code: int | None = None):
        super().__init__(f"[{service}] {message}")
        self.service = service
        self.status_code = status_code


class NasaConfigurationError(NasaApiError):
    """Credentials or settings required by a NASA service are missing."""
