import logging
import numpy as np

from src.converters.abstract_converter import AbstractConverter
from src.schemas.urg import URG


class Png2UrgConverter(AbstractConverter):
    def run(self, image: np.ndarray) -> URG:
        logging.info(f"Converting PNG to URG format")
        return URG()