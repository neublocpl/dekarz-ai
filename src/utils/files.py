import requests
import logging


def download_pdf_file(url: str) -> bytes:
    try:
        response = requests.get(url)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", "").lower():
            logging.warning(f"Content-Type for {url} is not 'application/pdf'.")

        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file from {url}: {e}")
        return None