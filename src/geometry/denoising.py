import cv2
import numpy as np
from src.geometry.pipeline import Pipeline, Tool


class FourierCleaner(Tool):
    def run(self, image, **kwargs):
        return denoise_with_fourier(image, **kwargs)


class Denoiser(Pipeline):
    DEFAULT_FLOW = []


def denoise_with_fourier(
    image,
    liczba_iteracji: int = 3,
    rozmiar_maski_centralnej: int | None = None,  # <<< ZWIĘKSZONA WARTOŚĆ
    sila_odszumiania: int = 25,
):
    obraz_przetwarzany = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rozmiar_maski_centralnej = (
        rozmiar_maski_centralnej or max(obraz_przetwarzany.shape) // 100
    )
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Krok 2: Główna pętla przetwarzania
    for i in range(liczba_iteracji):
        dft = cv2.dft(np.float32(obraz_przetwarzany), flags=cv2.DFT_COMPLEX_OUTPUT)

        dft_shift = np.fft.fftshift(dft)

        rows, cols = obraz_przetwarzany.shape
        crow, ccol = rows // 2, cols // 2

        maska = np.ones((rows, cols, 2), np.float32)
        r = int(rozmiar_maski_centralnej / 2)
        maska[crow - r : crow + r, ccol - r : ccol + r] = 0

        fshift_filtered = dft_shift * maska

        # Zapis spektrum
        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1
        )
        cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
        dft_shift_2 = np.fft.fftshift(dft_shift)

        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(dft_shift_2[:, :, 0], dft_shift_2[:, :, 1]) + 1
        )
        cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)

        dft_shift_2[magnitude_spectrum < magnitude_spectrum.max() * 2 / 5, :] = 0

        f_ishift_2 = np.fft.ifftshift(dft_shift_2)
        f_ishift_3 = np.fft.ifftshift(f_ishift_2)
        fshift_filtered = f_ishift_2
        obraz_po_fft_3 = cv2.idft(f_ishift_3)
        obraz_po_fft_3 = cv2.magnitude(obraz_po_fft_3[:, :, 0], obraz_po_fft_3[:, :, 1])

        # <<< DODANA INWERSJA OBRAZU ZGODNIE Z SUGESTIĄ

        cv2.normalize(obraz_po_fft_3, obraz_po_fft_3, 0, 255, cv2.NORM_MINMAX)

        f_ishift = np.fft.ifftshift(fshift_filtered)
        obraz_po_fft = cv2.idft(f_ishift)
        obraz_po_fft = cv2.magnitude(obraz_po_fft[:, :, 0], obraz_po_fft[:, :, 1])

        # <<< DODANA INWERSJA OBRAZU ZGODNIE Z SUGESTIĄ

        cv2.normalize(obraz_po_fft, obraz_po_fft, 0, 255, cv2.NORM_MINMAX)
        obraz_po_fft = np.uint8(obraz_po_fft)
        obraz_po_fft[obraz_po_fft > 128] = 255

        obraz_binaryzowany = cv2.adaptiveThreshold(
            obraz_po_fft,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        obraz_wyostrzony = cv2.filter2D(obraz_binaryzowany, -1, kernel_sharpening)
        obraz_odszumiony = cv2.fastNlMeansDenoising(
            obraz_wyostrzony,
            None,
            h=sila_odszumiania,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        obraz_przetwarzany = 255 - obraz_odszumiony

    return cv2.adaptiveThreshold(
        obraz_przetwarzany,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )
