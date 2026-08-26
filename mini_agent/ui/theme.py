"""theme.py — color system (a hand-tuned dark terminal palette)

Layering (three grays + semantic colors + brand color):
    text     white   — body/answers (brightest)
    inactive mid-gray — secondary info (think content, args, results)
    subtle   dark-gray — structural glyphs (⎿, counters, token stats) (darkest)
    claude   orange  — identity color (marker glyphs)
    each key color has a lighter shimmer twin — per-frame interpolation = flowing glow
"""
import random

# hand-tuned dark palette values
TEXT = "rgb(255,255,255)"
INACTIVE = "rgb(153,153,153)"
INACTIVE_SHIMMER = "rgb(193,193,193)"   # CC original: shimmer tier for streaming text
SUBTLE = "rgb(120,120,120)"       # CC original 80,80,80 is too dark on light terminals; bumped one notch
CLAUDE = "rgb(215,119,87)"        # claude orange
CLAUDE_SHIMMER = "rgb(235,159,127)"
SPIN = "rgb(147,165,255)"         # spinner blue
SPIN_SHIMMER = "rgb(177,195,255)"
SUCCESS = "rgb(78,186,101)"
ERROR = "rgb(255,107,128)"
WARNING = "rgb(255,193,7)"
WARNING_SHIMMER = "rgb(255,223,57)"

# spinner verbs (playful -ing verbs, picked at random)
SPINNER_VERBS = [
    "Pondering", "Cogitating", "Deliberating", "Musing", "Brewing",
    "Simmering", "Percolating", "Scheming", "Noodling", "Ruminating",
    "Conjuring", "Marinating", "Crunching", "Untangling", "Distilling",
]


def pick_verb() -> str:
    return random.choice(SPINNER_VERBS)


# ---------- flowing gradient (true shimmer: per-character phase wave) ----------
import math as _math

_RGB_RE = None

def _parse_rgb(s: str):
    global _RGB_RE
    if _RGB_RE is None:
        import re
        _RGB_RE = re.compile(r"rgb\((\d+),(\d+),(\d+)\)")
    m = _RGB_RE.match(s)
    return tuple(int(x) for x in m.groups())

def shimmer_text(s: str, frame: int, base: str = SPIN, light: str = SPIN_SHIMMER,
                 wavelength: float = 5.5, speed: float = 1.2, bold: bool = True):
    """Per-character glow: a sine wave travels along the text, interpolating base<->light.

    frame: external animation frame (12fps); wavelength: chars; speed: cycles/sec.
    Returns a rich Text.
    """
    from rich.text import Text
    b, l = _parse_rgb(base), _parse_rgb(light)
    t = Text()
    for i, ch in enumerate(s):
        phase = _math.sin(2 * _math.pi * (frame * speed / 12.0 - i / wavelength))
        k = (phase + 1) / 2
        r, g, bl = (round(b[j] + (l[j] - b[j]) * k) for j in range(3))
        style = f"rgb({r},{g},{bl})"
        t.append(ch, style=f"bold {style}" if bold else style)
    return t
