"""Shared chart style for the report: Persian text shaping (RTL), Persian digits, palette tokens."""
import re
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
from pathlib import Path
_FONTS = Path(__file__).resolve().parent.parent / "fonts"      # Vazirmatn v33.0.3 (OFL)
for f in ["Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf"]:
    font_manager.fontManager.addfont(str(_FONTS / f))
matplotlib.rcParams["font.family"] = "Vazirmatn"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal, validated (light)
FRAMES = [1379, 1392, 1397]                                          # sample-frame renewals (HBS)

_FA = str.maketrans("0123456789%", "۰۱۲۳۴۵۶۷۸۹٪")


def fa(text, digits=True):
    """Persian text for matplotlib >= 3.11 (it shapes and applies bidi itself); Persian digits."""
    # numeric ranges "25–29" are reordered by bidi (shown as 29–25); write them as "25 تا 29"
    s = re.sub(r"(\d+)\s*[–-]\s*(\d+)", r"\1 تا \2", str(text))
    return s.translate(_FA) if digits else s


def pct(x, nd=0):
    return fa(f"{x * 100:.{nd}f}%")


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))


def title(fig, main, sub=None):
    """Right-aligned Persian title block."""
    fig.text(0.985, 0.965, fa(main), ha="right", va="top", fontsize=15, color=INK, weight="bold")
    if sub:
        fig.text(0.985, 0.915, fa(sub), ha="right", va="top", fontsize=10.5, color=INK2)


def note(fig, lines):
    """Footnotes are not drawn on report charts (user preference); sources/methods live in Data/README.md."""
    return None
