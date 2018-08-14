import string
import os
import scipy as sp
import numpy as np
import scipy.linalg as spl
import time
import shutil

from pyezminc import read_xfm, write_xfm, xfm_to_param, param_to_xfm, \
    xfm_entry, xfm_param, xfm_identity, xfm_identity_transform_par


class XFM(object):
    """
    Class to deal with XFM files
    """
    def __init__(self, fname=None, xfm_matrix=None, displacement_volume=None, invert=True):
        """
        Returns an XFM object.
        Takes as input either an XFM filaneme (fname) or a data array (data).

        :param fname: Input filename (e.g. t1p-to-t2w.xfm)
        :type fname: str
        :param xfm_matrix: Input data array for the linear part of the transformation if any
        :type xfm_matrix: np.ndarray (4x4)
        :param displacement_volume: Input displacement minc volume for the non linear part of the transformation if any
        :type displacement_volume: basestring
        :param invert invert provided parameters
        :type boolean
        :return:
        """
        if fname and (xfm_matrix or displacement_volume):
            raise ValueError(
                "Input filename and data are both specified. "
                "The constructor needs either an input filename or input data array, but not both."
            )
        self.from_file = False
        self.fname = fname
        self.parameter_extracted = False
        self.parameter_changed   = False
        self.par = xfm_identity_transform_par()
        self.xfm = xfm_identity()
        self.history = []

        if self.fname:
            self.load(self.fname)
        else:
            if xfm_matrix:
                self.update_xfm_matrix(xfm_matrix,invert=invert)
            if displacement_volume:
                self.update_nl_deformation(displacement_volume,invert=invert)
                self.has_non_linear = True
                self.displacement_volume = displacement_volume

    def update_xfm_matrix(self, xfm_matrix=None):
        """
        Set based on affine transformation matrix
        :param xfm_matrix: 4x4 matrix
        :return:
        """
        if xfm_matrix:
            self.xfm = [ xfm_entry(True,False,np.asarray(xfm_matrix)) ]
            self.from_file = False
            self.parameter_extracted = False
            self.parameter_changed = False

    def update_nl_deformation(self,grid=None, invert=False):
        """
        Set non-linear transform given grid file
        :param grid: _grid file
        :param invert: inversion flag
        :return:
        """
        if grid:
            self.xfm = [xfm_entry(False, invert, grid)]
            self.from_file = False
            self.parameter_extracted = False
            self.parameter_changed  = False

    def load(self, input_filename):
        """
        Load from .xfm file
        :param input_filename: input filename
        :return:
        """
        if not os.path.exists(input_filename):
            raise IOError('file does not exist', input_filename)
        self.from_file = True
        self.xfm = read_xfm(input_filename)
        self.parameter_extracted = False
        self.parameter_changed = False

    def save(self, output_xfm_name, nl_invert=None):
        """
        Save the transformation.

        :param output_xfm_name: filename as which the transformation is saved
        """
        if self.parameter_changed:
            self._set_parameters()
        write_xfm(output_xfm_name, self.xfm, comments=self.history)


    @property
    def center(self):
        if not self.parameter_extracted:
            self._extract_parameters()
        return self.par.center

    @center.setter
    def center(self, value):
        """
        :param value: Center point for the linear part of the xfm
        :type value: np.array
        """
        self._center = value
        self.parameter_changed = True

    @property
    def translation(self):
        if not self.parameter_extracted:
            self._extract_parameters()
        return self.par.translation

    @translation.setter
    def translation(self, value):
        """
        :param value: Translation vector for the linear part of the xfm
        :type value: np.array
        """
        self.par.translation = value
        self.parameter_changed = True

    @property
    def rotation(self):
        if not self.parameter_extracted:
            self._extract_parameters()
        return self.par.rotation

    @rotation.setter
    def rotation(self, value):
        """
        :param value: Rotation vector for the linear part of the xfm
        :type value: np.array
        """
        self.par.rotation = value
        self.parameter_changed = True

    @property
    def scale(self):
        if not self.parameter_extracted:
            self._extract_parameters()
        return self.par.scale

    @scale.setter
    def scale(self, value):
        """
        :param value: Scale vector for the linear part of the xfm
        :type value: np.array
        """
        self.par.scale = value
        self.parameter_changed = True

    @property
    def shear(self):
        if not self.parameter_extracted:
            self._extract_parameters()
        return self.par.shear

    @shear.setter
    def shear(self, value):
        """
        :param value: Shear vector for the linear part of the xfm
        :type value: np.array
        """
        self.par.shear = value
        self.parameter_changed = True

    def append_history(self, content):
        h = [time.strftime('%a %b %d %H:%M:%S %Y'),
             '>>> ',
             content]
        self.history.append(string.join(h, sep=''))

    def sqrt(self):
        """
        Take square root of transform (half)
        :return: new transform
        """
        if self.parameter_changed:
            self._set_parameters()
        self._check_linear()
        assert(len(self.xfm)==1 and self.xfm[0].lin)
        return XFM(xfm_matrix = spl.sqrtm(self.xfm[0].trans).real)

    def inv(self):
        """
        Return inverted transform (linear only)
        :return:
        """
        if self.parameter_changed:
            self._set_parameters()
        self._check_linear()
        assert(len(self.par)==1 and self.par[0].lin)
        return XFM(xfm_matrix=spl.inv(self.matrix))

    def avg(self, xfms):
        """
        Average wotj arbitraru number of linear transforms
        :param xfms: another XFM object or list of XFMs
        :return: new XFM object
        """
        self._check_linear()

        if isinstance(xfms, XFM):
            xfms = [xfms]

        R = spl.logm(self.par[0].trans)
        for x in xfms:
            x._check_linear()
            R += spl.logm(x.par[0].trans)
        R/=len(xfms)+1
        R = spl.expm(R)
        R = sp.real(E)
        return XFM(xfm_matrix=R)

    def concat(self, xfms):
        """
        Concatenate with arbitrary xfm transforms
        :param xfms: a single or a list of transforms
        :return: new transform after concatenation
        """
        if self.parameter_changed:
            self._set_parameters()
        self._check_linear()
        if isinstance(xfms, XFM):
            xfms = [xfms]

        C = sp.identity(4)
        for x in reversed(xfms):
            x._check_linear()
            C = sp.dot(C, x.par[0].trans)
        C = sp.dot(C, self.par[0].trans)
        return XFM(xfm_matrix=C)

    def determinant(self):
        """ Return the determinant of the XFM array, assume linear transform """
        self._check_linear()
        return spl.det(self.matrix)

    def _set_parameters(self):
        self.xfm=param_to_xfm(self.par)
        self.parameter_changed = False
        self.parameter_extracted = True

    def _extract_parameters(self):
        self._check_linear()
        self.par=xfm_to_param(self.xfm)
        self.parameter_changed = False
        self.parameter_extracted = True

    def _check_linear(self):
        if len(self.xfm)!=1 or not self.xfm[0].lin:
            raise Exception("Only single linear transform is supported")
