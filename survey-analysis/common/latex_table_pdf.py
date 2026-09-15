"""Compile standalone LaTeX table fragments to PDF, PNG, or SVG (via tectonic)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

OutputFormat = Literal["pdf", "png", "svg", "pdf+svg", "svg+png", "both"]


def _trim_png_whitespace(png_path: Path, *, padding: int = 12) -> None:
    """Crop PNG to non-white content bounds."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return

    image = Image.open(png_path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if not bbox:
        return
    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(image.width, right + padding)
    lower = min(image.height, lower + padding)
    image.crop((left, upper, right, lower)).save(png_path)


def _pdf_to_png_pdftoppm(pdf_path: Path, png_path: Path, *, dpi: int) -> bool:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return False
    prefix = png_path.with_suffix("")
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            pdf_path.name,
            prefix.name,
        ],
        cwd=pdf_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    produced = pdf_path.parent / f"{prefix.name}.png"
    if produced != png_path:
        produced.replace(png_path)
    return png_path.is_file()


def _pdf_to_png_magick(pdf_path: Path, png_path: Path, *, dpi: int) -> bool:
    magick = shutil.which("magick") or shutil.which("convert")
    if not magick:
        return False
    subprocess.run(
        [magick, "-density", str(dpi), str(pdf_path), str(png_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return png_path.is_file()


def _pdf_to_png_qlmanage(pdf_path: Path, png_path: Path, *, max_edge: int) -> bool:
    """macOS Quick Look — renders LaTeX OTF fonts correctly (unlike sips)."""
    if sys.platform != "darwin" or not shutil.which("qlmanage"):
        return False
    out_dir = png_path.parent
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(max_edge), "-o", str(out_dir), str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    produced = out_dir / f"{pdf_path.name}.png"
    if not produced.is_file():
        return False
    if produced != png_path:
        produced.replace(png_path)
    return True


def _pdf_to_png_pymupdf(pdf_path: Path, png_path: Path, *, dpi: int) -> bool:
    try:
        import pymupdf
    except ImportError:
        return False
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
        pix.save(png_path)
    finally:
        doc.close()
    return png_path.is_file()


def _pdf_to_png(pdf_path: Path, png_path: Path, *, dpi: int = 300) -> None:
    """Rasterize a single-page PDF to PNG."""
    if _pdf_to_png_pdftoppm(pdf_path, png_path, dpi=dpi):
        return
    if _pdf_to_png_magick(pdf_path, png_path, dpi=dpi):
        return
    # qlmanage uses max edge pixels; ~3x letter width at 300dpi
    if _pdf_to_png_qlmanage(pdf_path, png_path, max_edge=max(2400, dpi * 8)):
        return
    if _pdf_to_png_pymupdf(pdf_path, png_path, dpi=dpi):
        return

    raise RuntimeError(
        "No working PDF→PNG renderer found. On macOS, qlmanage should be available; "
        "otherwise install poppler (pdftoppm), ImageMagick, or PyMuPDF."
    )


def _pdf_to_svg_pdftocairo(pdf_path: Path, svg_path: Path) -> bool:
    pdftocairo = shutil.which("pdftocairo")
    if not pdftocairo:
        return False
    subprocess.run(
        [pdftocairo, "-svg", pdf_path.name, svg_path.name],
        cwd=pdf_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return svg_path.is_file()


def _pdf_to_svg_pdf2svg(pdf_path: Path, svg_path: Path) -> bool:
    pdf2svg = shutil.which("pdf2svg")
    if not pdf2svg:
        return False
    subprocess.run(
        [pdf2svg, str(pdf_path), str(svg_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return svg_path.is_file()


def _pdf_to_svg_pymupdf(pdf_path: Path, svg_path: Path) -> bool:
    try:
        import pymupdf
    except ImportError:
        return False
    doc = pymupdf.open(pdf_path)
    try:
        if doc.page_count < 1:
            return False
        svg_path.write_text(doc[0].get_svg_image(), encoding="utf-8")
    finally:
        doc.close()
    return svg_path.is_file() and svg_path.stat().st_size > 0


def _pdf_to_svg(pdf_path: Path, svg_path: Path) -> None:
    """Convert a single-page PDF to SVG."""
    if _pdf_to_svg_pdftocairo(pdf_path, svg_path):
        return
    if _pdf_to_svg_pdf2svg(pdf_path, svg_path):
        return
    if _pdf_to_svg_pymupdf(pdf_path, svg_path):
        return
    raise RuntimeError(
        "No working PDF→SVG converter found. Install poppler (pdftocairo), "
        "pdf2svg, or PyMuPDF (pymupdf)."
    )


def compile_standalone_table(
    out_dir: Path,
    stem: str,
    body_tex: str,
    *,
    delete_intermediate_tex: bool = False,
    output_format: OutputFormat = "png",
    png_dpi: int = 300,
    extra_packages: list[str] | None = None,
    crop: Literal["preview", "standalone"] = "preview",
    font: Literal["newtx", "helvetica"] = "newtx",
) -> Path:
    """Write ``stem.tex`` + ``stem_standalone.tex``, compile, return PDF/PNG/SVG path.

    ``crop="standalone"`` uses the standalone document class so the PDF is
    tightly cropped to the table (avoids full-textwidth caption blank space).

    ``font="helvetica"`` uses Helvetica (``helvet``) as the default text font,
    matching Nature-style matplotlib figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / f"{stem}.tex"
    standalone_path = out_dir / f"{stem}_standalone.tex"
    pdf_path = out_dir / f"{stem}_standalone.pdf"
    png_path = out_dir / f"{stem}_standalone.png"
    svg_path = out_dir / f"{stem}_standalone.svg"

    body_path.write_text(body_tex, encoding="utf-8")

    if font == "helvetica":
        font_packages = [
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[scaled]{helvet}",
            "\\renewcommand{\\familydefault}{\\sfdefault}",
            "\\usepackage[italic]{mathastext}",
        ]
    else:
        font_packages = [
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{newtxtext,newtxmath}",
        ]

    packages = [
        *font_packages,
        "\\usepackage[table]{xcolor}",
        "\\usepackage{colortbl}",
        "\\usepackage{booktabs}",
        "\\usepackage{multirow}",
        "\\usepackage{arydshln}",
        "\\usepackage{caption}",
        "\\captionsetup{font=small,labelfont=bf}",
    ]
    if extra_packages:
        packages.extend(extra_packages)

    if crop == "standalone":
        standalone = "\n".join([
            f"% Auto-generated standalone wrapper for {stem}.tex",
            f"% Compile: tectonic {stem}_standalone.tex",
            "\\documentclass[border=4pt]{standalone}",
            *packages,
            "",
            "\\begin{document}",
            f"\\input{{{stem}.tex}}",
            "\\end{document}",
            "",
        ])
    else:
        standalone = "\n".join([
            f"% Auto-generated standalone wrapper for {stem}.tex",
            f"% Compile: tectonic {stem}_standalone.tex",
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=0pt]{geometry}",
            "\\usepackage[active,tightpage]{preview}",
            *packages,
            "",
            "\\begin{document}",
            "\\begin{preview}",
            "",
            f"\\input{{{stem}.tex}}",
            "",
            "\\end{preview}",
            "\\end{document}",
            "",
        ])
    standalone_path.write_text(standalone, encoding="utf-8")

    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError(
            "tectonic not found; install with: brew install tectonic"
        )
    result = subprocess.run(
        [tectonic, standalone_path.name],
        cwd=out_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tectonic failed ({standalone_path.name}):\n{result.stderr or result.stdout}"
        )
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF not produced: {pdf_path}")

    if output_format == "pdf":
        if delete_intermediate_tex:
            body_path.unlink(missing_ok=True)
            standalone_path.unlink(missing_ok=True)
        return pdf_path

    if output_format in {"svg", "pdf+svg", "svg+png", "both"}:
        _pdf_to_svg(pdf_path, svg_path)
        if not svg_path.is_file():
            raise RuntimeError(f"SVG not produced: {svg_path}")
        if output_format == "svg+png":
            _pdf_to_png(pdf_path, png_path, dpi=png_dpi)
            if not png_path.is_file():
                raise RuntimeError(f"PNG not produced: {png_path}")
            _trim_png_whitespace(png_path)
            pdf_path.unlink(missing_ok=True)
            if delete_intermediate_tex:
                body_path.unlink(missing_ok=True)
                standalone_path.unlink(missing_ok=True)
            return png_path
        keep_pdf = output_format in {"pdf+svg", "both"}
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)
        if delete_intermediate_tex:
            body_path.unlink(missing_ok=True)
            standalone_path.unlink(missing_ok=True)
        return svg_path if not keep_pdf else pdf_path

    _pdf_to_png(pdf_path, png_path, dpi=png_dpi)
    if not png_path.is_file():
        raise RuntimeError(f"PNG not produced: {png_path}")

    _trim_png_whitespace(png_path)
    pdf_path.unlink(missing_ok=True)

    if delete_intermediate_tex:
        body_path.unlink(missing_ok=True)
        standalone_path.unlink(missing_ok=True)

    return png_path


# Backward-compatible alias
compile_standalone_table_pdf = compile_standalone_table
