import os
import numpy as np

from src.converters import Pdf2PngConverter, Png2UrgConverter
from src.utils import debug
from src.utils.files import download_pdf_file
from src.utils.errors import DownloadError, ConversionError
from src.schemas.urg import URG


class MainPipeline:
    def __init__(self):
        self._pdf_to_png_converter = Pdf2PngConverter()
        self._png_to_urg_converter = Png2UrgConverter()

    def run(self, file_url: str, job_uuid: str) -> URG:
        debug.set_current_job_id(job_uuid)

        pdf_data = download_pdf_file(file_url)
        if pdf_data is None:
            raise DownloadError(f"Failed to download PDF from {file_url}")

        png_data = self._pdf_to_png_converter.run(pdf_data)
        if png_data is None:
            raise ConversionError("Failed to convert PDF to PNG")

        urg_data = self._png_to_urg_converter.run(png_data)
        if urg_data is None:
            raise ConversionError("Failed to convert PNG to URG")

        return urg_data
