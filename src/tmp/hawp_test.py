#!/usr/bin/env python3
"""
run_hawp_on_input.py

Użycie:
    python run_hawp_on_input.py --input dokument.pdf \
        --ckpt checkpoints/hawpv3-imagenet-03a84.pth \
        --out results \
        --threshold 0.05

Obsługiwane przypadki:
 - pojedynczy obraz (jpg/png/itp.)
 - plik PDF (każda strona konwertowana do PNG)
 - katalog z obrazami

We wszystkich przypadkach HAWP dostaje katalog z plikami PNG.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200):
    """Konwertuje PDF do PNG (jedna strona -> jeden plik)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    out_paths = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_file = out_dir / f"page_{i:04d}.png"
        pix.save(str(out_file))
        out_paths.append(out_file)
    doc.close()
    return out_paths


def is_image_file(p: Path):
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


# def collect_images_from_input(input_path: Path, tmp_dir: Path, dpi: int = 200) -> Path:
#     """
#     Zwraca ścieżkę do folderu z obrazami, które trzeba podać do HAWP.
#     Wszystko jest normalizowane tak, żeby HAWP zawsze dostawał folder.
#     """
#     tmp_dir.mkdir(parents=True, exist_ok=True)
#
#     if input_path.is_dir():
#         # folder z obrazami
#         return input_path
#
#     if input_path.suffix.lower() == ".pdf":
#         # PDF -> konwersja do PNG w tmp_dir
#         images = pdf_to_images(input_path, tmp_dir, dpi=dpi)
#         if not images:
#             raise RuntimeError(f"PDF {input_path} nie wygenerował żadnych stron!")
#         return tmp_dir
#
#     if is_image_file(input_path):
#         # pojedynczy obraz -> kopiujemy do tmp_dir
#         dest = tmp_dir / input_path.name
#         shutil.copy2(input_path, dest)
#         return tmp_dir
#
#     raise ValueError(f"Nieobsługiwany typ pliku wejściowego: {input_path}")


def run_hawp_predict(img_files: list[Path], ckpt_path: str, saveto: str,
                     ext: str = "png", threshold: float = 0.05,
                     device: str = "cpu", extra_args=None):
    """
    Uruchamia moduł predykcji HAWP (CLI).
    img_files: lista ścieżek do plików (nie katalog!)
    """
    if not img_files:
        raise ValueError("Brak plików wejściowych dla HAWP!")

    cmd = [
        sys.executable, "-m", "hawp.ssl.predict",
        "--ckpt", ckpt_path,
        "--img", *[str(f) for f in img_files],
        "--saveto", saveto,
        "--ext", ext,
        "--threshold", str(threshold),
        "--device", device,
    ]
    if extra_args:
        cmd += extra_args

    print("Uruchamiam HAWP:", " ".join(cmd))
    subprocess.check_call(cmd)


def collect_images_from_input(input_path: Path, tmp_dir: Path, dpi: int = 200) -> list[Path]:
    """
    Zwraca listę plików obrazowych do podania w --img.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if is_image_file(p)])

    if input_path.suffix.lower() == ".pdf":
        return pdf_to_images(input_path, tmp_dir, dpi=dpi)

    if is_image_file(input_path):
        dest = tmp_dir / input_path.name
        shutil.copy2(input_path, dest)
        return [dest]

    raise ValueError(f"Nieobsługiwany typ pliku wejściowego: {input_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Ścieżka do obrazu, PDF, lub folderu z obrazami.")
    p.add_argument("--ckpt", required=True, help="Ścieżka do checkpointu .pth (HAWP).")
    p.add_argument("--out", default="hawp_out", help="Folder, gdzie zapisać wyniki.")
    p.add_argument("--threshold", type=float, default=0.05, help="Próg detekcji (domyślnie 0.05).")
    p.add_argument("--dpi", type=int, default=200, help="DPI przy konwersji PDF -> PNG.")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--keep_tmp", action="store_true", help="Zachowaj tymczasowy folder z obrazami.")
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    tmp_dir = Path(".hawp_tmp_images")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # przygotuj obrazy do analizy
        img_dir = collect_images_from_input(input_path, tmp_dir, dpi=args.dpi)

        # upewnij się, że katalog wyjściowy istnieje
        out_dir.mkdir(parents=True, exist_ok=True)

        # uruchom HAWP
        run_hawp_predict(img_dir, args.ckpt, str(out_dir),
                         ext="png", threshold=args.threshold, device=args.device)

        print(f"Gotowe. Wyniki zapisane w: {out_dir.resolve()}")
    finally:
        if args.keep_tmp:
            print(f"Zachowano tymczasowe obrazy: {tmp_dir.resolve()}")
        else:
            pass
            # shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
