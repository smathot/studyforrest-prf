#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import nibabel as nib


class PrfMap:

    def __init__(self, data, bold):

        self._data = data
        self._affine = bold.affine

    def __getitem__(self, key):

        return self._data[key]

    def __setitem__(self, key, value):

        self._data[key] = value

    def keep(self, x=None, y=None, sd=None, err=None):

        if x is not None:
            self._data[self._data[:, :, :, 0] < x[0]] = np.nan
            self._data[self._data[:, :, :, 0] > x[1]] = np.nan
        if y is not None:
            self._data[self._data[:, :, :, 1] < y[0]] = np.nan
            self._data[self._data[:, :, :, 1] > y[1]] = np.nan
        if sd is not None:
            self._data[self._data[:, :, :, 2] < sd[0]] = np.nan
            self._data[self._data[:, :, :, 2] > sd[1]] = np.nan
        if err is not None:
            self._data[self._data[:, :, :, 3] < err[0]] = np.nan
            self._data[self._data[:, :, :, 3] > err[1]] = np.nan

    @property
    def params(self):

        x = self.x.flatten()
        y = self.y.flatten()
        sd = self.sd.flatten()
        err = self.err.flatten()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        sd = sd[~np.isnan(sd)]
        err = err[~np.isnan(err)]
        return x, y, sd, err

    @property
    def x(self):

        return self._data[:, :, :, 0]

    @property
    def y(self):

        return self._data[:, :, :, 1]

    @property
    def sd(self):

        return self._data[:, :, :, 2]

    @property
    def err(self):

        return self._data[:, :, :, 3]

    @property
    def ximg(self):

        return self._img(0)

    @property
    def yimg(self):

        return self._img(1)

    @property
    def sdimg(self):

        return self._img(2)

    @property
    def errimg(self):

        return self._img(3)

    def _img(self, i):

        return nib.Nifti1Image(self._data[:, :, :, i], self._affine)
