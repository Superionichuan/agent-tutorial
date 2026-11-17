"""Pixel art of the pets, drawn by the code in this file.

The dolphin is made of a few shapes: a body along a curved line (the back arched, the tail
curled under), a beak, a dorsal fin, a flipper, tail flukes and an eye. Each pixel is
sampled 4 x 4 times; it is filled when most samples fall inside, and coloured by where
they fall: the back in blues (darker toward the outline), the belly pale, the tail grey
blue. No image, font or emoji is used. Frames: the dolphin, one pixel higher, again.
"""
import math

W, H = 18, 22                       # pixels; the renderer shows two pixel rows per character
B36 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"

DARK, MID, LIGHT, SHINE = (16, 74, 94), (22, 128, 168), (52, 186, 222), (120, 232, 250)
BELLY, BELLY_SHADE, TAIL, TAIL_DARK, EYE = (214, 232, 234), (164, 184, 190), (112, 148, 158), (78, 104, 112), (10, 30, 40)

TORSO = (8.9, 6.4, 7.5, 3.9)        # the body: centre x, y and half axes of an ellipse
TAIL_LINE = ((14.2, 8.4), (16.8, 15.6), (9.2, 17.6))   # the tail: a curve, body to flukes


def _tail(t):
    """A point of the tail's centre line, t from 0 (at the body) to 1 (at the flukes)."""
    (x0, y0), (x1, y1), (x2, y2) = TAIL_LINE
    u = 1 - t
    return u * u * x0 + 2 * u * t * x1 + t * t * x2, u * u * y0 + 2 * u * t * y1 + t * t * y2


def _in_triangle(p, a, b, c):
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def _near_segment(p, a, b, width):
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (vx * vx + vy * vy)))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy) <= width


TAIL_POINTS = [(i / 100, _tail(i / 100)) for i in range(101)]
BEAK = ((2.6, 8.9), (0.3, 9.9))
FIN = ((10.6, 3.0), (14.8, 4.2), (16.6, 0.8))
FLIPPER = ((5.8, 9.9), (8.4, 9.6), (8.8, 12.7))
FLUKES = ((9.8, 17.4), (6.4, 14.6), (6.0, 20.6))
EYE_AT = (4.3, 6.4)


def _colour(x, y):
    """The colour at a point of the picture, or None outside the dolphin."""
    if math.hypot(x - EYE_AT[0], y - EYE_AT[1]) <= 0.6:
        return EYE
    cx, cy, a, b = TORSO
    e = math.hypot((x - cx) / a, (y - cy) / b)
    if e <= 1:
        v = (y - cy) / b                                   # -1 top of the back, +1 belly
        if v > 0.45 and x < cx + 3.5:
            return BELLY if v < 0.85 else BELLY_SHADE
        if e > 0.8:
            return DARK if v < -0.2 or x > cx + 3.5 else MID
        return SHINE if v < -0.1 and x < cx + 2.5 else LIGHT
    d, t = min((math.hypot(x - px, y - py), t) for t, (px, py) in TAIL_POINTS)
    if d <= 2.5 - 1.7 * t:
        px, py = _tail(t)
        qx, qy = _tail(min(1.0, t + 0.01))
        outer = (qx - px) * (y - py) - (qy - py) * (x - px) < 0       # the right-hand side
        if t > 0.55:
            return TAIL_DARK if outer else TAIL
        return (DARK if d > 0.5 * (2.5 - 1.7 * t) else MID) if outer else TAIL
    if _near_segment((x, y), *BEAK, 0.9):
        return LIGHT if y < 9.3 else BELLY
    if _in_triangle((x, y), *FIN):
        return DARK
    if _in_triangle((x, y), *FLIPPER):
        return TAIL
    if _in_triangle((x, y), *FLUKES):
        return TAIL_DARK if y > 18.8 or y < 15.6 else TAIL
    return None


def _draw():
    palette, rows = [], []
    for y in range(H):
        row = ""
        for x in range(W):
            votes = {}
            for sy in range(4):
                for sx in range(4):
                    c = _colour(x + (sx + 0.5) / 4, y + (sy + 0.5) / 4)
                    if c is not None:
                        votes[c] = votes.get(c, 0) + 1
            if sum(votes.values()) < 7:
                row += "."
                continue
            colour = max(votes, key=lambda c: (votes[c], c == EYE))
            if colour not in palette:
                palette.append(colour)
            row += B36[palette.index(colour)]
        rows.append(row)
    up = rows[1:] + ["." * W]
    return {"palette": palette, "frames": [rows, up, rows]}


PETS_PIXEL = {"dolphin": _draw()}
