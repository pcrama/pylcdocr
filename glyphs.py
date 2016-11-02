import numpy

def scale_array(arr, scale):
    if (scale != int(scale)) or (scale < 1):
        raise RuntimeError("scale={!r} must be a positive integer".format(scale))
    elif scale == 1:
        return arr
    if len(arr.shape) == 2:
        result = numpy.zeros(
            (arr.shape[0] * scale, arr.shape[1] * scale),
            dtype=arr.dtype)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                for sub_row in range(scale):
                    for sub_col in range(scale):
                        result[row * scale + sub_row, col * scale + sub_col] = arr[row, col]
    elif len(arr.shape) == 3:
        result = numpy.zeros(
            (arr.shape[0] * scale, arr.shape[1] * scale, arr.shape[2]),
            dtype=arr.dtype)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                for sub_row in range(scale):
                    for sub_col in range(scale):
                        for component in range(arr.shape[2]):
                            result[row * scale + sub_row,
                                   col * scale + sub_col,
                                   component] = arr[row, col, component]
    else:
        raise NotImplementedError("Only 2D greyscale or 2D color image "
                                  "scaling is implemented")
    return result

def colorize_greyscale(arr, color, bg_color=None):
    basetype = numpy.uint8
    try:
        result = numpy.zeros((arr.shape[0], arr.shape[1], len(color)),
                             dtype=basetype)
    except TypeError:
        # color is a scalar (can't be enumerated)
        if bg_color is None:
            bg_color = 0
        result = numpy.zeros((arr.shape[0], arr.shape[1]),
                             dtype=basetype)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                    result[row, col] = basetype(round(
                        arr[row, col] * (color - bg_color) + bg_color))
    else:
        # color can be enumerated
        if bg_color is None:
            bg_color = (0, 0, 0)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                for (component, (value, bg_value)) in enumerate(zip(color, bg_color)):
                    result[row, col, component] = basetype(round(
                        arr[row, col] * (value - bg_value) + bg_value))
    return result

def render_array_on_img(arr, img, x, y):
    if len(arr.shape) != len(img.shape):
        raise RuntimeError("arr and img must have same color type")
    elif (len(arr.shape) == 3) and (arr.shape[2] != img.shape[2]):
        raise RuntimeError("arr and img must have same color depth")
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            if (0 <= (col + x) < img.shape[1]) and (
                    0 <= (row + y) < img.shape[0]):
                if len(arr.shape) == 3:
                    for component in range(len(arr.shape)):
                        img[row + y, col + x, component] = arr[row, col, component]
                else:
                    img[row + y, col + x] = arr[row, col]

class Glyph(object):
    def __init__(self, array):
        self.data = {1: array,}
        self.height = array.shape[0]
        self.width = array.shape[1]
    def scaled(self, scale):
        try:
            return self.data[scale]
        except KeyError:
            self.data[scale] = scale_array(self.data[1], scale)
            return self.data[scale]
    def render(self, img, x, y, scale=1, color=None, bg_color=None):
        return render_array_on_img(
            self.scaled(scale) if color is None
            else colorize_greyscale(self.scaled(scale), color, bg_color),
            img,
            x,
            y)


# 4 pixels below base line + 7 pixels for lower case + 3 pixels for ascenders
FONT = {
    "a": Glyph(numpy.array((
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 1, 1, 1, 0, ),
        (0, 0, 0, 0, 1, ),
        (0, 0, 0, 0, 1, ),
        (0, 1, 1, 1, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (0, 1, 1, 1, 1, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),))),
    "b": Glyph(numpy.array((
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 0, ),
        (1, 1, 1, 1, 0, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 1, 1, 1, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),))),
    "c": Glyph(numpy.array((
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 1, 1, 1, 0, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 1, ),
        (0, 1, 1, 1, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),))),
    "d": Glyph(numpy.array((
        (0, 0, 0, 0, 1, ),
        (0, 0, 0, 0, 1, ),
        (0, 0, 0, 0, 1, ),
        (0, 1, 1, 1, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (0, 1, 1, 1, 1, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),))),
    "e": Glyph(numpy.array((
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 1, 1, 1, 0, ),
        (1, 0, 0, 0, 1, ),
        (1, 0, 0, 0, 1, ),
        (1, 1, 1, 1, 1, ),
        (1, 0, 0, 0, 0, ),
        (1, 0, 0, 0, 0, ),
        (0, 1, 1, 1, 1, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),
        (0, 0, 0, 0, 0, ),))),
    "i": Glyph(numpy.array((
        (0, 0, 0, ),
        (0, 1, 0, ),
        (0, 0, 0, ),
        (1, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (1, 1, 1, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),))),
    "t": Glyph(numpy.array((
        (0, 0, 0, ),
        (0, 1, 0, ),
        (1, 1, 1, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 0, 1, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),))),
    "l": Glyph(numpy.array((
        (1, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (0, 1, 0, ),
        (1, 1, 1, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),
        (0, 0, 0, ),))),
    " ": Glyph(numpy.array((
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),
        (0, 0, 0, 0, ),))),
}

def render_string_on_img(str_, img, x, y, scale=1, color=None, font=FONT, line_spacing=1, glyph_spacing=1, bg_color=None):
    start_x = x
    max_h = 0
    for c in str_:
        if c == '\n':
            y += max_h + line_spacing
            x = start_x
        else:
            glyph = font[c]
            glyph.render(img, x, y, scale, color, bg_color=bg_color)
            max_h = max(max_h, glyph.height * scale)
            x += glyph.width * scale + glyph_spacing
    return img
