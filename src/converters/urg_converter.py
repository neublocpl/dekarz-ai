from abc import ABC, abstractmethod
from typing import Any

from src.schemas import URG


class AbstractUrgConverter(ABC):
    @abstractmethod
    def run(self, input_data: Any) -> URG:
        raise NotImplementedError("Subclasses must implement this method")