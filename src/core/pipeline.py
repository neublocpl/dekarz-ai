import os
import numpy as np

from src.converters import Png2UrgConverter
from src.schemas.urg import URG


class MainPipeline:
    def __init__(self):
        pass

    def run(self, file_url: str) -> URG:
        # Download the file from the URL
        # IF PDF -> convert to PNG
        urg_data = Png2UrgConverter().run(image=np.zeros((100, 100))) # Placeholder
        # Do some post-processing

        return urg_data
