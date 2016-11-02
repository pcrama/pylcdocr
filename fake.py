"""Draw fake pictures to test detection"""

import random

import cv2
import numpy

import glyphs

def fake(img_dim, screen_dim, text, edge_tol=None, black=(3, 3, 3), color=(128, 128, 128), bg_color=(250, 250, 250), with_smoothing=True, with_noise=True):
    result = numpy.zeros((img_dim[0], img_dim[1], 3), dtype="uint8")
    scale = min(int(img_dim[0] / screen_dim[0]), int(img_dim[1] / screen_dim[1]))
    vert_border = (img_dim[0] - screen_dim[0] * scale) / 2
    horz_border = (img_dim[1] - screen_dim[1] * scale) / 2
    # define four corners
    ul = (horz_border + (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)),
          vert_border + (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)))
    ur = (img_dim[1] - horz_border +
          (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)),
          vert_border + (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)))
    bl = (horz_border + (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)),
          img_dim[0] - vert_border +
          (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)))
    br = (img_dim[1] - horz_border +
          (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)),
          img_dim[0] - vert_border +
          (0 if edge_tol is None else random.randint(-edge_tol, edge_tol)))
    # draw black border
    cv2.fillPoly(result,
                 [numpy.array(((0, 0), ul, ur,
                               br, bl, ul,
                               (0, 0), (img_dim[1] - 1, 0),
                               (img_dim[1] - 1, img_dim[0] - 1),
                               (0, img_dim[0] - 1),
                               (0, 0))
                 )],
                 black)
    # fill screen with white
    cv2.fillPoly(result,
                 [numpy.array((ul, ur, br, bl, ul,))],
                 bg_color)
    # write string
    result = glyphs.render_string_on_img(text, result,
                                         horz_border + ((2 * scale) if edge_tol is None else edge_tol),
                                         vert_border + ((2 * scale) if edge_tol is None else edge_tol),
                                         scale=scale,
                                         color=color,
                                         bg_color=bg_color,
                                         line_spacing=scale,
                                         glyph_spacing=scale)
    if with_smoothing:
        # smooth
        result = cv2.blur(result, ((5 * scale + 1) / 4, (5 * scale + 1) / 4))
    if with_noise:
        # add noise: where ever the random values are larger than a treshold,
        # replace the pixel by its inverse
        noise = numpy.empty(result.shape[:-1], dtype=numpy.float)
        cv2.randu(noise, 0.0, 16.0 * float(scale))
        noise = noise <= 1.0
        # make noise 3D (to have same value across RGB channels
        noise = numpy.swapaxes(numpy.array((noise, noise, noise)).T, 0, 1)
        result = numpy.multiply(result, (numpy.uint8(1) - noise)) + numpy.multiply(
            numpy.uint8(255) - result, noise)
    return result
