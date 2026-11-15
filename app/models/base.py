from abc import ABC, abstractmethod


class BaseDocumentDetector(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes):
        raise NotImplementedError


class BaseOCRProvider(ABC):
    @abstractmethod
    def extract_text(self, image_bytes: bytes):
        raise NotImplementedError


class BaseDefectDetector(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes):
        raise NotImplementedError

