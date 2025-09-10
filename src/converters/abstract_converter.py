from abc import ABC, abstractmethod

from src.schemas.urg import URG


class AbstractConverter(ABC):
    @abstractmethod
    def run(self) -> URG:
        raise NotImplementedError("Subclasses must implement this method")