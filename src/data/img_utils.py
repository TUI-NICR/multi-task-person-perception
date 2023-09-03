# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
from operator import attrgetter

import cv2
import numpy as np


def _const(*args):
    """
    Return constant depending on OpenCV version.
    Returns first value found for supplied names of constant.
    """
    for const in args:
        try:
            return attrgetter(const)(cv2)
        except AttributeError:
            continue
    raise AttributeError(
        """Installed OpenCV version {:s} has non of the given constants.
         Tested constants: {:s}""".format(cv2.__version__, ', '.join(args))
    )


def _rint(value):
    """Round and convert to int"""
    return int(np.round(value))


# interpolation modes (all supported)
_INTERPOLATION_DICT = {
    # bicubic interpolation
    'bicubic': _const('INTER_CUBIC', 'INTER_CUBIC'),
    # nearest-neighbor interpolation
    'nearest': _const('INTER_NEAREST', 'INTER_NEAREST'),
    # bilinear interpolation (4x4 pixel neighborhood)
    'linear': _const('INTER_LINEAR', 'INTER_LINEAR'),
    # resampling using pixel area relation, preferred for shrinking
    'area': _const('INTER_AREA', 'INTER_AREA'),
    # Lanczos interpolation (8x8 pixel neighborhood)
    'lanczos4': _const('INTER_LANCZOS4', 'INTER_LANCZOS4')
}


def check_dtype(dtype, allowed_dtypes):
    """
    Function to check if a given dtype is a subdtype of one of the dtypes in a
    given list.

    Parameters
    ----------
    dtype : {numpy.dtype, type}
        The dtype to check.
    allowed_dtypes : {numpy.dtype, list, tuple}
        Allowed dtype or list of allowed dtypes.

    Returns
    -------
    is_subdtype : bool
        True, if the given dtype is a valid subdtype, otherwise False.

    """
    # ensure that allowed_dtypes is a list of dtypes
    if not isinstance(allowed_dtypes, list):
        if isinstance(allowed_dtypes, tuple):
            allowed_dtypes = list(allowed_dtypes)
        else:
            allowed_dtypes = [allowed_dtypes]

    # np.bool is deprecated in numpy > 1.20
    # see: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    # fix bool type, since np.issubdtype(any_type, bool) is always true
    # see: https://github.com/numpy/numpy/issues/5711
    # try:
    #     idx = allowed_dtypes.index(np.bool)
    #     allowed_dtypes[idx] = np.bool_
    # except ValueError:
    #     # np.bool/bool is not in allowed_dtypes
    #     pass

    # check dtype
    for allowed_dtype in allowed_dtypes:
        if np.issubdtype(dtype, allowed_dtype):
            return True
    return False


def load(filepath, mode=None):
    if not os.path.exists(filepath):
        raise IOError("No such file or directory: '{}'".format(filepath))

    if mode is None:
        mode = cv2.IMREAD_UNCHANGED
    img = cv2.imread(filepath, mode)

    if img is None:
        print("ERROR Reading Image. Possibly corrupted PGM File:")
        print(filepath)
    if img.ndim > 2:
        if img.shape[-1] == 4:
            color_mode = cv2.COLOR_BGRA2RGBA
        else:
            color_mode = cv2.COLOR_BGR2RGB

        img = cv2.cvtColor(img, color_mode)
    return img


def resize(img, shape_or_scale, interpolation='linear'):
    """
    Function to resize a given image.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.
    shape_or_scale : {float, tuple, list}
        The output image shape as a tuple of ints (height, width), the scale
        factors for both dimensions as a tuple of floats (fy, fx) or a single
        float as scale factor for both dimensions.
    interpolation : str
        Interpolation method to use, one of: 'nearest', 'linear' (default),
        'area', 'bicubic' or 'lanczos4'. For details, see OpenCV documentation.

    Returns
    -------
    img_resized : numpy.ndarray
        The resized input image.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16, np.float32))

    # get current shape
    cur_height, cur_width = img.shape[:2]

    # check shape_or_scale
    if isinstance(shape_or_scale, (tuple, list)) and len(shape_or_scale) == 2:
        if all(isinstance(e, int) for e in shape_or_scale):
            new_height, new_width = shape_or_scale
        elif all(isinstance(e, float) for e in shape_or_scale):
            fy, fx = shape_or_scale
            new_height = _rint(fy*cur_height)
            new_width = _rint(fx*cur_width)
        else:
            raise ValueError("`shape_or_scale` should either be a tuple of "
                             "ints (height, width) or a tuple of floats "
                             "(fy, fx)")
    elif isinstance(shape_or_scale, float):
        new_height = _rint(shape_or_scale * cur_height)
        new_width = _rint(shape_or_scale * cur_width)
    else:
        raise ValueError("`shape_or_scale` should either be a tuple of ints "
                         "(height, width) or a tuple of floats (fy, fx) or a "
                         "single float value")

    # scale image
    if cur_height == new_height and cur_width == new_width:
        return img

    return cv2.resize(img,
                      dsize=(new_width, new_height),
                      interpolation=_INTERPOLATION_DICT[interpolation])
