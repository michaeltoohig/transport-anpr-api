def convert_hex_rgb(_hex: str) -> dict:
    _hex = _hex.lstrip("#")
    j = len(_hex)
    values = tuple(int(_hex[i : i + j // 3], 16) for i in range(0, j, j // 3))
    return dict(r=values[0], g=values[1], b=values[2])


def convert_rgb_hex(rgb: dict) -> str:
    return "{0:02x}{1:02x}{2:02x}".format(rgb["r"], rgb["g"], rgb["b"])


def convert_rgb_xyz(rgb: dict) -> dict:
    """
    Convert RGB to XYZ colour space.

    An intermediate step to convert RGB to LAB.
    """
    r = rgb["r"] / 255
    g = rgb["g"] / 255
    b = rgb["b"] / 255
    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92
    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92
    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92
    r = r * 100
    g = g * 100
    b = b * 100
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return dict(x=X, y=Y, z=Z)

def convert_rgb_lab(rgb: dict) -> dict:
    """
    Convert RGB to LAB colour space.

    Reference X, Y, Z refer to standard whitepoint values.
    """
    whitepoints = dict(
        baseline=(1.0000, 1.0000, 1.0000),  # theoretical reference point
        d50=(0.9642, 1.0000, 0.8251),  # warn sunrise / sunset daylight
        d55=(0.9568, 1.0000, 0.9214),  # mid-morning / mid-afternoon daylight
        d65=(0.9504, 1.0000, 1.0888),  # noon daylight
    )
    ref_x, ref_y, ref_z = whitepoints.get("d65")

    xyz = convert_rgb_xyz(rgb)
    x = xyz.get("x") / ref_x
    y = xyz.get("y") / ref_y
    z = xyz.get("z") / ref_z
    if x > 0.008856:
        x = x ** (1 / 3)
    else:
        x = (7.787 * x) + (16 / 116)
    if y > 0.008856:
        y = y ** (1 / 3)
    else:
        y = (7.787 * y) + (16 / 116)
    if z > 0.008856:
        z = z ** (1 / 3)
    else:
        z = (7.787 * z) + (16 / 116)
    L = (116 * y) - 16
    A = 500 * (x - y)
    B = 200 * (y - z)
    return dict(l=L, a=A, b=B)


def convert_rgb_hsl(rgb: dict) -> dict:
    """
    Convert RGB to HSL colour space.

    Ported from colorsys.py
    """
    maxc = max(rgb[key] for key in rgb.keys())
    minc = min(rgb[key] for key in rgb.keys())
    # XXX Can optimize (maxc + minc) and (maxc - minc)
    L = (minc + maxc) / 2.0
    if minc == maxc:
        return dict(h=0.0, s=0.0, l=L)
    if L <= 0.5:
        S = (maxc - minc) / (maxc + minc)
    else:
        S = (maxc - minc) / (2.0 - maxc - minc)
    rc = (maxc - rgb["r"]) / (maxc - minc)
    gc = (maxc - rgb["g"]) / (maxc - minc)
    bc = (maxc - rgb["b"]) / (maxc - minc)
    if rgb["r"] == maxc:
        H = bc - gc
    elif rgb["g"] == maxc:
        H = 2.0 + rc - bc
    else:
        H = 4.0 + gc - rc
    H = (H / 6.0) % 1.0
    return dict(h=H, s=S, l=L)

def distance(c1, c2) -> int:
    """
    Return the difference between two colour objects by their LAB values.
    """
    return (
        (c1.lab["l"] - c2.lab["l"]) ** 2 * 0.7
        + (c1.lab["a"] - c2.lab["a"]) ** 2
        + (c1.lab["b"] - c2.lab["b"]) ** 2
    ) ** 0.5
