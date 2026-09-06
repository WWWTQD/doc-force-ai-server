from typing import Any


class AppException(Exception):
    """Base application exception carrying a code, message and HTTP status."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class InvalidFileError(AppException):
    def __init__(self, message: str = "Invalid file", details: Any = None):
        super().__init__("INVALID_FILE", message, status_code=400, details=details)


class UnsupportedFormatError(AppException):
    def __init__(self, message: str = "Unsupported format", details: Any = None):
        super().__init__("UNSUPPORTED_FORMAT", message, status_code=400, details=details)


class FileTooLargeError(AppException):
    def __init__(self, message: str = "File too large", details: Any = None):
        super().__init__("FILE_TOO_LARGE", message, status_code=413, details=details)


class InvalidImageError(AppException):
    def __init__(self, message: str = "Unable to decode image", details: Any = None):
        super().__init__("INVALID_IMAGE", message, status_code=400, details=details)


class ProcessingError(AppException):
    def __init__(self, message: str = "Processing error", details: Any = None):
        super().__init__("PROCESSING_ERROR", message, status_code=500, details=details)

