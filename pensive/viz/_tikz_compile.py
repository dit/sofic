"""Compile Vaucanson TikZ fragments to PDF/PNG/SVG via ``pdflatex``."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_VAUCANSON_ASSET = Path(__file__).resolve().parent / "assets" / "vaucanson.tikz"

# GUI-launched notebooks often omit MacTeX / Homebrew from PATH.
_TOOL_SEARCH_DIRS = (
    Path("/Library/TeX/texbin"),
    Path("/usr/local/texlive/2026/bin/universal-darwin"),
    Path("/usr/local/texlive/2025/bin/universal-darwin"),
    Path("/usr/local/texlive/2024/bin/universal-darwin"),
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


class TikzCompileError(RuntimeError):
    """Raised when LaTeX compilation or image conversion fails."""


def find_executable(name: str) -> str | None:
    """Locate a CLI tool on PATH or in common TeX/Homebrew install dirs."""
    found = shutil.which(name)
    if found is not None:
        return found
    for directory in _TOOL_SEARCH_DIRS:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _require_pdflatex() -> str:
    path = find_executable("pdflatex")
    if path is None:
        raise TikzCompileError(
            "TikZ image export requires pdflatex (e.g. MacTeX or TeX Live). "
            "Install TeX and ensure pdflatex is on PATH, typically "
            "/Library/TeX/texbin on macOS."
        )
    return path


def compilation_document(body: str) -> str:
    """Wrap a ``tikzpicture`` fragment in a standalone LaTeX document."""
    return (
        r"\documentclass[tikz,border=2pt]{standalone}"
        "\n"
        r"\usepackage{amsmath}"
        "\n"
        r"\usepackage{xcolor}"
        "\n"
        r"\definecolor{honeydew}{RGB}{240,255,240}"
        "\n"
        r"\definecolor{mistyrose}{RGB}{255,228,225}"
        "\n"
        r"\usepackage{nicefrac}"
        "\n"
        r"\providecommand{\half}{\nicefrac{1}{2}}"
        "\n"
        r"\usetikzlibrary{automata,positioning,arrows.meta}"
        "\n"
        r"\input{vaucanson.tikz}"
        "\n"
        r"\begin{document}"
        "\n"
        f"{body}\n"
        r"\end{document}"
        "\n"
    )


def compile_tikz_fragment(fragment: str, *, format: str = "png") -> bytes:
    """Compile a ``tikzpicture`` fragment and return raster/vector bytes."""
    normalized = format.lower().lstrip(".")
    if normalized not in {"pdf", "png", "svg"}:
        raise ValueError(f"unsupported compile format {format!r}; expected pdf, png, or svg")

    pdflatex = _require_pdflatex()
    with tempfile.TemporaryDirectory(prefix="pensive-tikz-") as tmp:
        workdir = Path(tmp)
        shutil.copy2(_VAUCANSON_ASSET, workdir / "vaucanson.tikz")
        tex_path = workdir / "pensive_tikz.tex"
        tex_path.write_text(compilation_document(fragment), encoding="utf-8")

        result = subprocess.run(
            [pdflatex, "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
        pdf_path = workdir / "pensive_tikz.pdf"
        if result.returncode != 0 or not pdf_path.is_file():
            log_tail = (workdir / "pensive_tikz.log").read_text(encoding="utf-8", errors="replace")
            raise TikzCompileError("pdflatex failed to compile TikZ figure.\n" + log_tail[-4000:])

        if normalized == "pdf":
            return pdf_path.read_bytes()
        if normalized == "png":
            return _pdf_to_png(pdf_path, workdir / "pensive_tikz.png")
        return _pdf_to_svg(pdf_path, workdir / "pensive_tikz.svg")


def _pdf_to_png(pdf_path: Path, png_path: Path) -> bytes:
    pdftocairo = find_executable("pdftocairo")
    if pdftocairo is not None:
        subprocess.run(
            [pdftocairo, "-png", "-singlefile", str(pdf_path), str(png_path.with_suffix(""))],
            check=True,
            capture_output=True,
        )
        if png_path.is_file():
            return png_path.read_bytes()

    gs = find_executable("gs")
    if gs is not None:
        subprocess.run(
            [
                gs,
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pngalpha",
                "-r200",
                "-dFirstPage=1",
                "-dLastPage=1",
                f"-sOutputFile={png_path}",
                str(pdf_path),
            ],
            check=True,
            capture_output=True,
        )
        if png_path.is_file():
            return png_path.read_bytes()

    if os.uname().sysname == "Darwin":
        sips = find_executable("sips")
        if sips is not None:
            subprocess.run([sips, "-s", "format", "png", str(pdf_path), "--out", str(png_path)], check=True)
            if png_path.is_file():
                return png_path.read_bytes()

    raise TikzCompileError(
        "compiled PDF but could not convert to PNG; install poppler (pdftocairo) or ghostscript (gs)"
    )


def _pdf_to_svg(pdf_path: Path, svg_path: Path) -> bytes:
    pdftocairo = find_executable("pdftocairo")
    if pdftocairo is not None:
        subprocess.run(
            [pdftocairo, "-svg", str(pdf_path), str(svg_path)],
            check=True,
            capture_output=True,
        )
        if svg_path.is_file():
            return svg_path.read_bytes()

    pdf2svg = find_executable("pdf2svg")
    if pdf2svg is not None:
        subprocess.run([pdf2svg, str(pdf_path), str(svg_path)], check=True, capture_output=True)
        if svg_path.is_file():
            return svg_path.read_bytes()

    raise TikzCompileError("compiled PDF but could not convert to SVG; install poppler (pdftocairo) or pdf2svg")
