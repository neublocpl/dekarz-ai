import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
import os
from pathlib import Path


def finalne_przetwarzanie_z_detekcja_linii(
        # sciezka_pdf: str = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf",
        sciezka_pdf: str = "ex1.pdf",
        numer_strony: int = 0,
        liczba_iteracji: int = 3,
        rozmiar_maski_centralnej: int = 35,  # <<< ZWIĘKSZONA WARTOŚĆ
        sila_odszumiania: int = 25
):
    """
    Funkcja w pętli przetwarza obraz, a na końcu wykrywa na nim linie proste.
    Proces: FFT -> Filtracja -> Inwersja -> Binaryzacja -> Wyostrzenie -> Odszumienie.
    Po pętli: Detekcja linii metodą Hougha i wizualizacja.

    Args:
        sciezka_pdf (str): Ścieżka do pliku PDF.
        numer_strony (int): Numer strony do przetworzenia.
        liczba_iteracji (int): Liczba powtórzeń całego procesu.
        rozmiar_maski_centralnej (int): Średnica maski do usunięcia widma centralnego.
        sila_odszumiania (int): Parametr filtra Non-Local Means.
    """
    pref = Path(sciezka_pdf).stem
    print(pref)
    try:
        # Krok 1: Wczytanie i przygotowanie obrazu
        obrazy = convert_from_path(sciezka_pdf)
        if not obrazy or numer_strony >= len(obrazy):
            print(f"Błąd: Nie można wczytać strony {numer_strony + 1} z pliku '{sciezka_pdf}'.")
            return

        obraz_pil = obrazy[numer_strony]
        obraz_przetwarzany = cv2.cvtColor(np.array(obraz_pil), cv2.COLOR_RGB2GRAY)

        # obraz_przetwarzany = cv2.adaptiveThreshold(
        #     obraz_przetwarzany, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY_INV, 11, 2
        # )

        rozmiar_maski_centralnej = max(obraz_przetwarzany.shape) // 100
        print(rozmiar_maski_centralnej)

        oryginalny_obraz = obraz_przetwarzany.copy()
        cv2.imwrite("oryginal.jpg", oryginalny_obraz)
        print("Zapisano oryginalny obraz jako 'oryginal.jpg'")

        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        # Krok 2: Główna pętla przetwarzania
        for i in range(liczba_iteracji):
            print(f"\n--- Rozpoczęcie iteracji {i + 1}/{liczba_iteracji} ---")

            dft = cv2.dft(np.float32(obraz_przetwarzany), flags=cv2.DFT_COMPLEX_OUTPUT)
            print(dft.shape)
            dft_shift = np.fft.fftshift(dft)

            rows, cols = obraz_przetwarzany.shape
            crow, ccol = rows // 2, cols // 2

            maska = np.ones((rows, cols, 2), np.float32)
            r = int(rozmiar_maski_centralnej / 2)
            maska[crow - r:crow + r, ccol - r:ccol + r] = 0

            fshift_filtered = dft_shift * maska

            # Zapis spektrum
            magnitude_spectrum = 20 * np.log(cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1)
            cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(f"iteracja_{i + 1}_spektrum.jpg", magnitude_spectrum)
            print(dft_shift.shape)
            dft_shift_2 = np.fft.fftshift(dft_shift)

            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift_2[:, :, 0], dft_shift_2[:, :, 1]) + 1)
            cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite(f"iteracja_{i + 1}_dsa.jpg", magnitude_spectrum)
            print("B")
            print(magnitude_spectrum.shape, magnitude_spectrum.max(), magnitude_spectrum.min(), type(magnitude_spectrum))
            dft_shift_2[magnitude_spectrum < magnitude_spectrum.max() * 2/5, :] = 0

            f_ishift_2 = np.fft.ifftshift(dft_shift_2)
            f_ishift_3 = np.fft.ifftshift(f_ishift_2)
            fshift_filtered = f_ishift_2
            obraz_po_fft_3 = cv2.idft(f_ishift_3)
            obraz_po_fft_3 = cv2.magnitude(obraz_po_fft_3[:, :, 0], obraz_po_fft_3[:, :, 1])

            # <<< DODANA INWERSJA OBRAZU ZGODNIE Z SUGESTIĄ

            cv2.normalize(obraz_po_fft_3, obraz_po_fft_3, 0, 255, cv2.NORM_MINMAX)
            obraz_po_fft_3 = np.uint8(obraz_po_fft_3)

            cv2.imwrite(f"{pref}_krok_{i + 1}_xyz.png", obraz_po_fft_3)

            f_ishift = np.fft.ifftshift(fshift_filtered)
            obraz_po_fft = cv2.idft(f_ishift)
            obraz_po_fft = cv2.magnitude(obraz_po_fft[:, :, 0], obraz_po_fft[:, :, 1])

            # <<< DODANA INWERSJA OBRAZU ZGODNIE Z SUGESTIĄ

            cv2.normalize(obraz_po_fft, obraz_po_fft, 0, 255, cv2.NORM_MINMAX)
            obraz_po_fft = np.uint8(obraz_po_fft)
            obraz_po_fft[obraz_po_fft > 128] = 255

            obraz_binaryzowany = cv2.adaptiveThreshold(
                obraz_po_fft, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            obraz_wyostrzony = cv2.filter2D(obraz_binaryzowany, -1, kernel_sharpening)
            obraz_odszumiony = cv2.fastNlMeansDenoising(obraz_wyostrzony, None, h=sila_odszumiania,
                                                        templateWindowSize=7, searchWindowSize=21)

            obraz_przetwarzany = 255 - obraz_odszumiony
            cv2.imwrite(f"{pref}_krok_{i + 1}b.png", obraz_przetwarzany)
            print(f"Zapisano finalny obraz z iteracji {i + 1} jako 'krok_{i + 1}.png'")
        return obraz_przetwarzany
        # Krok 3: Wykrywanie linii na finalnym obrazie
        print("\n--- Rozpoczęcie detekcji linii ---")

        # Konwertujemy finalny obraz (który jest w skali szarości) na format BGR, aby móc rysować kolorowe linie
        finalny_obraz_kolor = cv2.cvtColor(obraz_przetwarzany, cv2.COLOR_GRAY2BGR)

        # Używamy Progresywnej Probabilistycznej Transformacji Hougha
        # Parametry (można je dostosować):
        # threshold: Minimalna liczba "głosów" (punktów na prostej), aby uznać ją za linię.
        # minLineLength: Minimalna długość linii w pikselach. Krótsze segmenty są odrzucane.
        # maxLineGap: Maksymalna dozwolona przerwa między segmentami, aby traktować je jako jedną linię.
        linie = cv2.HoughLinesP(obraz_przetwarzany, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

        if linie is not None:
            print(f"Wykryto {len(linie)} linii.")
            for linia in linie:
                x1, y1, x2, y2 = linia[0]
                # Rysujemy czerwoną linię (kolor BGR: 0, 0, 255) o grubości 2 pikseli
                cv2.line(finalny_obraz_kolor, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            print("Nie wykryto żadnych linii spełniających kryteria.")

        # Zapisujemy obraz z naniesionymi liniami
        # cv2.imwrite("wynik_z_liniami.jpg", finalny_obraz_kolor)
        print("Zapisano obraz z wykrytymi liniami jako 'wynik_z_liniami.jpg'")

        # Krok 4: Końcowa wizualizacja
        # plt.figure(figsize=(12, 8))
        # plt.imshow(cv2.cvtColor(finalny_obraz_kolor, cv2.COLOR_BGR2RGB))
        # plt.title(f'Finalny wynik z wykrytymi liniami')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

    except FileNotFoundError:
        print(f"Błąd: Plik '{sciezka_pdf}' nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


# --- Uruchomienie ---
# finalne_przetwarzanie_z_detekcja_linii()
# files_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow"
#
# start = 1
# stop = 69
# # methods = {1: methode1, 2: methode2}
# methode = 2
# # chosen_methode = methods[methode]
# dump_path = f"./outputs_{methode}"
# os.makedirs(dump_path, exist_ok=True)
#
# for path_id in range(start, stop):
#     finalne_przetwarzanie_z_detekcja_linii(
#         f"{files_path}/{path_id}.pdf"
#     )
#     break