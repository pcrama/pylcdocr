import math
import os

import cv2
import numpy

import glyphs

def optional_display(img, win_name, default_win_name):
    if win_name is not None:
        win_name = (win_name
                    if isinstance(win_name, basestring)
                    else default_win_name)
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

def load_image(fin, winname=None):
    img = cv2.imread(fin)
    optional_display(img, winname, os.path.splitext(os.path.basename(fin))[0])
    return img

def draw_line_in_polar_coords(img, rho, theta, color, thickness=None):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    cv2.line(img,
             (int(round(x0 - 1000 * b)), int(round(y0 + 1000 * a))),
             (int(round(x0 + 1000 * b)), int(round(y0 - 1000 * a))),
             color,
             thickness=thickness)

def xy_intersect_of_polar_coords(rho1, theta1, rho2, theta2):
    """(x, y) coordinates of 2 lines defined by their parameters in polar coordinates

    @param rho1: distance from origin to line 1
    @param theta1: angle in radians between horizontal line and perpendicular to line 1
    @param rho2: distance from origin to line 2
    @param theta2: angle in radians between horizontal line and perpendicular to line 2

    @note: This function assumes that the lines cross and doesn't take special
    care to be numerically stable.  Its intended usage is to straighten a
    rectangle that has only be slightly warped by the perspective of the
    camera taking its picture.

        >>> xy_intersect_of_polar_coords(0, 0, 0, math.pi / 2)
        (0.0, 0.0)
        >>> xy_intersect_of_polar_coords(0, math.pi/4, math.sqrt(2), - math.pi / 4)
        (1.0, -1.0000000000000002)
    """
    # Represent line 1 as a*x + b*y = e
    a = math.cos(theta1)
    b = math.sin(theta1)
    e = rho1
    # Represent line 2 as c*x + d*y = f
    c = math.cos(theta2)
    d = math.sin(theta2)
    f = rho2
    # These two lines cross in x, y where x, y is the solution of the 2 linear
    # equations defined above.
    det = a * d - b * c
    x = (d * e - b * f) / det
    y = (a * f - c * e) / det
    return x, y

def detect_frame(img, winname=None, approx_scale=8):
    # smoothed = cv2.bilateralFilter(img, max(approx_scale / 2, 2), 10, max(approx_scale / 2, 2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    optional_display(gray, winname, "gray")
    # TODO: make this depend on ratio between image size and frame size.  Goal
    # is to make single (double?) pixel lines in original frame disappear such
    # that text drawn inside the screen won't influence the edge detection.
    DILATION_EROSION_ITERATIONS = approx_scale
    kernel = numpy.ndarray((0, ), numpy.uint8)
    dilated = cv2.dilate(gray,
                         kernel,
                         dst=numpy.ndarray(img.shape[:-1], numpy.uint8),
                         iterations=DILATION_EROSION_ITERATIONS,
                         borderType=cv2.BORDER_CONSTANT,
                         borderValue=0)
    # After dilation, use erosion to grow the black border more or less back
    # into its original place, otherwise, the final frame would be
    # 2*DILATION_EROSION_ITERATIONS image pixels too big.
    eroded = cv2.erode(dilated,
                       kernel,
                       dst=numpy.ndarray(img.shape[:-1], numpy.uint8),
                       iterations=DILATION_EROSION_ITERATIONS,
                       borderType=cv2.BORDER_CONSTANT,
                       borderValue=0)
    optional_display(dilated, winname, "dilated")
    optional_display(eroded, winname, "eroded")
    edges = cv2.Canny(eroded, 50, 150,
                      apertureSize=3, # kernel size of Sobel operator for gradient
    )
    optional_display(edges, winname, "edges")
    lines = cv2.HoughLines(edges, 3, math.pi / 180, int(round(min(img.shape[:-1]) / 2.5)))
    # Cluster lines, assuming that the picture rotation is smaller than
    # MAX_ROTATION and that the frame to detect fills more than 1/2 of the
    # display (the same assumption is also embedded in the threshold used in
    # the Hough transform).
    MAX_ROTATION = 20 * math.pi / 180 # angle in radians
    left, right, top, bottom = [], [], [], []
    for lin in lines:
        rho, theta = lin[0]
        # 0 < theta < \pi with -\infty < rho < \infty after Hough transform.
        # We will want to take averages, so nearly vertical lines (theta
        # closer to either 0 or \pi than to \pi/2) should be normalized to
        # -\pi/4 < theta < \pi/4 and rho > 0.
        appendee = None
        if (theta < MAX_ROTATION) or ((math.pi - theta) < MAX_ROTATION):
            # vertical line
            if theta > MAX_ROTATION:
                theta -= math.pi
                rho = -rho
            if rho < img.shape[1] / 4:
                appendee = left
            elif rho > img.shape[1] * 3 / 4:
                appendee = right
        elif abs(theta - math.pi / 2) < MAX_ROTATION:
            # horizontal line
            if rho < img.shape[0] / 4:
                appendee = top
            elif rho > img.shape[0] * 3 / 4:
                appendee = bottom
        if appendee is not None:
            appendee.append((rho, theta))
        else:
            print ("?", rho, theta)
    means = []
    for lines, color in (
            (left, (255, 0, 0)),
            (top, (255, 0, 0)),
            (right, (0, 255, 0)),
            (bottom, (0, 0, 255)),
    ):
        rho = theta = 0
        for lin in lines:
            rho += lin[0]
            theta += lin[1]
        if winname:
            draw_line_in_polar_coords(img, rho / len(lines), theta / len(lines), color, 1)
        means.append((rho / len(lines), theta / len(lines)))
    corners = list(xy_intersect_of_polar_coords(rho, theta,
                                                means[(idx + 1) % len(means)][0], means[(idx + 1) % len(means)][1])
               for (idx, (rho, theta)) in enumerate(means))
    if winname:
        for x, y in corners:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 255, 0))
        optional_display(img, winname, "lines")
    return numpy.float32(corners)

def fn_or_array_to_array(fn, imread_flags=None):
    if isinstance(fn, basestring):
        return (load_image(fn, None)
                if imread_flags is None else
                cv2.imread(fn, imread_flags))
    elif hasattr(fn, "__array__"):
        return fn
    else:
        raise TypeError("{!r} is neither a string nor a valid image".format(fn))

def straighten(fn, winname, width=640, height=480):
    img = fn_or_array_to_array(fn)
    approx_scale = int(min(float(img.shape[0]) / height,
                           float(img.shape[1]) / width))
    smoothed = cv2.medianBlur(img, (approx_scale / 2) * 2 + 1, dst=None)
    optional_display(img, winname, "smoothed")
    corners = detect_frame(smoothed.copy(), winname, approx_scale=approx_scale)
    xform = cv2.getPerspectiveTransform(
        corners,
        numpy.float32([(0,0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]))
    return cv2.warpPerspective(smoothed, xform, (width, height))

def detect_glyph(fn, glyph_fn, winname, width=640, height=480):
    img = straighten(fn,
                     "straighten: {}".format(winname) if winname else winname,
                     width=width,
                     height=height)
    (thresh, im_bw) = cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        128, # ignored if OTSU's method is used
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    optional_display(im_bw, winname, "B/W")
    import pdb; pdb.set_trace()
    glyph = fn_or_array_to_array(glyph_fn, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(im_bw / 255, numpy.uint8(1 - glyph), method=cv2.TM_SQDIFF)
    result8 = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    optional_display(glyphs.scale_array(result8, 5),
                     winname,
                     "{}: {}".format(os.path.basename(fn), os.path.basename(glyph_fn))
                     if isinstance(glyph_fn, basestring) else "Glyph")
    min_val, max_val, min_x_y, max_x_y = cv2.minMaxLoc(result)
    im_bw_bgr = glyphs.colorize_greyscale(im_bw / 255, (255, 255, 255), bg_color=(0, 0, 0))
    cv2.rectangle(im_bw_bgr, (min_x_y[0], min_x_y[1]), (min_x_y[0] + glyph.shape[1], min_x_y[1] + glyph.shape[0]), (255, 0, 0), 1)
    cv2.rectangle(im_bw_bgr, (min_x_y[0] - 1, min_x_y[1] - 1), (min_x_y[0], min_x_y[1]), (0, 255, 0), 1)
    cv2.rectangle(im_bw_bgr, (min_x_y[0] + glyph.shape[1], min_x_y[1] + glyph.shape[0]), (min_x_y[0] + glyph.shape[1] + 1, min_x_y[1] + glyph.shape[0] + 1), (0, 0, 255), 1)
    optional_display(im_bw_bgr, "A", "B")
    return (im_bw, glyph, result, result8, (min_val, max_val, min_x_y, max_x_y))

def get_local_minima(img):
    local_minima = []
    for row in range(img.shape[0]):
        min_row_delta = -1 if row > 0 else 0
        max_row_delta = 1 if row < (img.shape[0] - 1) else 0
        for col in range(img.shape[1]):
            min_col_delta = -1 if col > 0 else 0
            max_col_delta = 1 if col < (img.shape[1] - 1) else 0
            local_min = img[row, col]
            for row_delta in range(min_row_delta, max_row_delta + 1, 1):
                for col_delta in range(min_col_delta, max_col_delta + 1, 1):
                    if (row_delta != 0) or (col_delta != 0):
                        if local_min > img[row + row_delta, col + col_delta]:
                            local_min = None
                            break
                if local_min is None:
                    break
            if local_min is not None:
                local_minima.append((local_min, row, col))
    return local_minima

# (im_bw, glyph, result, result8, min_max_loc) = reload(straighten).detect_glyph(fake.fake((535, 673), (128, 160), "    a\n\n\n\n    aibei", edge_tol=1, with_smoothing=False, with_noise=False), glyphs.FONT["b"].scaled(2), "A", width=320, height=256)
