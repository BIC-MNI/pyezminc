#  Copyright 2013, Haz-Edine Assemlal

#  This file is part of PYEZMINC.
# 
#  PYEZMINC is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 2.
# 
#  PYEZMINC is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with PYEZMINC.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import subprocess as sp
from operator import mul
import numpy as np
import scipy.ndimage as ndimage
from numpy.core.numeric import indices
import os
import copy
import re
from datetime import datetime
from time import gmtime, strftime
import tempfile
import string
import shutil
import sys

from . import pyezminc


FIRST_LABEL_ID = 1

def entropy(histData):
    """ Compute entropy using the histogram of the data.
    
    :params array histData: The values of the histogram
    :returns: Will return entropy.
    """

    ps = histData/float(np.sum(histData))  # coerce to float and normalize
    ps = ps[np.nonzero(ps)]         # toss out zeros
    H = -np.sum(ps * np.log2(ps))      # compute entropy
    
    return H

def mi(x, y, bins, vRange):
    """Compute mutual information between two arrays
    
    :params array x: Input data
    :params array y: Input data
    :params array_like, shape(2,2) vRange: The leftmost and rightmost edges of 
                                           the bins for 2D histogram computation
                                           [[xmin, xmax], [ymin, ymax]]
    :returns: Will return mutual information value.
    """
    
    hist_xy = np.histogram2d(x, y, bins=bins, range=vRange)[0]
    hist_x  = np.histogram(x, bins=bins, range=vRange[0])[0]
    hist_y  = np.histogram(y, bins=bins, range=vRange[1])[0]
    
    H_xy = entropy(hist_xy)
    H_x  = entropy(hist_x)
    H_y  = entropy(hist_y)
    
    return H_x + H_y - H_xy

def gzip(fname=None, verbose=False):
    """ Gzip a file

    :param string fname: filename to gzip
    :returns: the filename of the gzipped file
    :rtype: string
    """
    sp.check_call(['gzip', '-f', fname])
    return fname + '.gz'

def mincinfo(header, field):
    """Gets data from a dictionary using a dotted accessor-string"""
    h = header
    for chunk in field.split(':'):
        h = h.get(chunk, {})
    if h == {}:
        raise Exception("field not found in header: '{0}'".format(field))
    return h

def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))

def display(images, label=None, imitate=None):
    ''' Display MINC images. The files are temporary saved to disk.
    :param images: minc.Image or list of minc.Image 
    :param label: minc.Label
    :param imitate: Which image header to imitate when saving images.
    '''
    if not isinstance(images, (list, set)):
        images = [images]
    tmp_images = dict()
    tmpdir = tempfile.mkdtemp(prefix='tmp_Display_')
    for f in images:
        if not f.name:
            tmpfname = system.create_tmp_filename(remove=False)
        else:
            tmpfname = os.path.join(tmpdir, f.name.basename())
        tmp_images[tmpfname] = f
    if label:
        tmp_label = system.create_tmp_filename(prefix='tmp_'+label.name.basename(), remove=False)
    try:
        for fname, img in tmp_images.iteritems():
            img.save(name=fname, imitate=imitate)
        cmd = 'Display {0}'.format(string.join(tmp_images.keys()))

        if label:
            label.save(name=tmp_label, imitate=imitate)
            cmd += ' -label {0}'.format(tmp_label)
        print(cmd)
        system.check_call_interact(cmd)
    except OSError:
        # Strange bug with pexpect which throws an exception when quitting
        pass
    finally:
        to_remove = tmp_images.keys()
        if label:
            to_remove += [tmp_label]
        for f in tmp_images.iterkeys():
            if os.path.isfile(f):
                os.remove(f)
        shutil.rmtree(tmpdir)


class Image(object):
    '''Class to deal with minc files. The data is stored
    in the data attribute as a numpy array. '''

    def __init__(self, fname=None, data=None, autoload=True, dtype=np.float64, *args, **kwargs):
        self.dtype = dtype
        if pyezminc is None:
            raise ImportError("pyezminc not found")

        self.__wrapper = pyezminc.EZMincWrapper()
        self.history = ''
        self.direction_cosines = None
        self.data = data
        self.name = fname

        if autoload and data is None and self.name:
            self.load(self.name, *args, **kwargs)

        if fname:
            self.history = self.__wrapper.history().rstrip('\n').split('\n')
            self.direction_cosines = self._get_direction_cosines()

    def __getattr__(self, name):
        if name == 'data':
            return self.__wrapper.data
        else:
            raise AttributeError("'{0}' object has no attribute '{1}'"\
                                 .format(type(self), name))

    def __setattr__(self, name, value):
        if name == 'data':
            self.__wrapper.data = value
        else:
            super(Image, self).__setattr__(name, value)

    def load(self, name=None, *args, **kwargs):
        if name is None:
            name = self.name
        if not os.path.exists(name):
            raise IOError('file does not exist', name)
        self._load(name, *args, **kwargs)

    def _load(self, name, with_nan=False, metadata_only=False, *args, **kwargs):
        '''Load a file'''
        self.__wrapper.load(name, dtype=self.dtype, metadata_only=metadata_only)
        self.header = self._read_header(name)
        # catch NaN values
        if not metadata_only and np.any(self.data == -sys.float_info.max):
            if not with_nan:
                raise Exception("NaN value detected in '{0}'".format(name))
            else:
                self.data = np.where(self.data == -sys.float_info.max, np.nan, self.data).astype(self.dtype)

    def save(self, name=None, *args, **kwargs):
        if name is None:
            name = self.name
        self._save(name, *args, **kwargs)

    def _save(self, name=None, imitate=None, history=None, *args, **kwargs):
        '''Save the image to a file.

        Args
            imitate: a reference filename for the header
        '''
        if not imitate:
            if not self.name:
                raise Exception('imitate or name options have to be defined')
            imitate = self.name

        if not os.path.isfile(imitate):
            raise Exception("Cannot imitate from non existing file: '{0}'".format(imitate))

        compress = False
        if os.path.splitext(name)[1] == '.gz':
            name = os.path.splitext(name)[0]
            compress = True

        if self.dtype != self.data.dtype:
            raise Exception("Cannot save image because non consistent dtype, '{0}' != '{1}'".format(self.dtype, self.data.dtype))
            
        if history is not None:
            self.history=history

        self.__wrapper.save(name, imitate=imitate, dtype=self.dtype, history=self.history )

        if compress:
            gzip(name)

    def _read_header(self, filename):
        if not self.__wrapper:
            raise Exception('not loaded from a file')
        try:
            header = self.__wrapper.parse_header()
        except Exception as e:
            print("MINC header exception to be fixed: {0}".format(e))
            header = None
        return header
    
    def get_MINC_header(self):
        return self.header

    def _save_reader(self):
        raise Exception('not yet implemented')

    def spacing(self):
        return [self.__wrapper.nspacing(i) for i in range(1, self.__wrapper.nb_dim()+1)]
    
    def dim(self):
        return [d for d in reversed(self.data.shape)]
    
    def start(self):
        return [self.__wrapper.nstart(i) for i in range(1, self.__wrapper.nb_dim()+1)]

    def volume(self, nb_voxels):
        '''Given a number of voxels, it returns the volume.'''
        one_voxel_cc = [self.__wrapper.nspacing(i)
                        for i in range(1,self.__wrapper.nb_dim()+1)]
        return nb_voxels * reduce(mul, one_voxel_cc)

    def _get_direction_cosines(self):
        dimensions = ('xspace', 'yspace', 'zspace')
        cosines = {}
        for i, d in enumerate(dimensions):
            if self.__wrapper.have_dir_cos(i+1):
                cosines[d] = tuple(self.__wrapper.ndir_cos(i+1,j) for j in range(3))
            else:
                cosines[d] = tuple(1 if j==i else 0 for j in range (3))
        return cosines
    
    def voxel_to_world(self, voxel):
        dimensions = ('xspace', 'yspace', 'zspace')
        world_tmp = [voxel[i]*self.spacing()[i] + self.start()[i] for i in range(3)]
        _dir_cos = self._get_direction_cosines()
        cosines = tuple(_dir_cos[d] for d in dimensions)
        cosines_transpose = np.transpose(np.asarray(cosines))
        world = []
        for i in range(3):
            world.append(sum(p*q for p,q in zip(world_tmp, cosines_transpose[i]))) 
        return world

    def scanner_manufacturer(self, normalize=True):
        extract = mincinfo(self.header, 'study:manufacturer')
        if not normalize:
            return extract
        valid = ['siemens', 'ge', 'philips', 'toshiba', 'hitachi', 'picker', 'marconi']
        match = np.array([self._match_header(v, extract) for v in valid])
        if np.sum(match) > 1:
            raise Exception('multiple match for scanner manufacturer')
        if np.sum(match) < 1:
            raise Exception('no match for scanner manufacturer')
        return valid[np.argmax(match)]

    def _match_header(self, regexp, value):
        return bool(re.search(regexp, value, re.IGNORECASE))
        
    def is_scanner_siemens(self):
        return self._match_header('siemens', self.scanner_manufacturer(normalize=False))

    def is_scanner_ge(self):
        return self._match_header('ge', self.scanner_manufacturer(normalize=False))

    def is_scanner_philips(self):
        return self._match_header('philips', self.scanner_manufacturer(normalize=False))

    def is_scanner_toshiba(self):
        return self._match_header('toshiba', self.scanner_manufacturer(normalize=False))

    def is_scanner_hitachi(self):
        return self._match_header('hitachi', self.scanner_manufacturer(normalize=False))

    def is_scanner_marconi(self):
        return self._match_header('marconi', self.scanner_manufacturer(normalize=False))
        
    def is_scanner_picker(self):
        return self._match_header('picker', self.scanner_manufacturer(normalize=False))

    def field_strength(self, field='dicom_0x0018:el_0x0087'):
        out = mincinfo(self.header, field)
        val = re.search('([0-9\.]*)'.format(field), out).group(1)
        # Make sure this is a number
        try:
            float(val)
        except:
            raise Exception('Could not extract the field strength as a float value: "{0}"'.format(val))
        return float(val)

    def acquisition_date(self):

       #::StartAcquiring[Acq-Time ]--->Acquiring--->::StartStoring[ContentTime]-->Storing.    
        #(0008,0013) TM [171809.637000]                          #  14, 1 InstanceCreationTime
        #(0008,0020) DA [20111006]                               #   8, 1 StudyDate
        #(0008,0021) DA [20111006]                               #   8, 1 SeriesDate
        #(0008,0022) DA [20111006]                               #   8, 1 AcquisitionDate
        #(0008,0023) DA [20111006]                               #   8, 1 ContentDate
        #(0008,0030) TM [165732.988000]                          #  14, 1 StudyTime
        #(0008,0031) TM [171609.633000]                          #  14, 1 SeriesTime
        #(0008,0032) TM [171736.510000]                          #  14, 1 AcquisitionTime
        #(0008,0033) TM [171809.637000]                          #  14, 1 ContentTime

       field_date = ['dicom_0x0008:el_0x0020',
                     'dicom_0x0008:el_0x0021',
                     'dicom_0x0008:el_0x0022',
                     'dicom_0x0008:el_0x0023',
                     'dicom_0x0008:el_0x0012']
       field_time = ['dicom_0x0008:el_0x0032',
                     'dicom_0x0008:el_0x0031',
                     'dicom_0x0008:el_0x0033']
       field_datetime = ['acquisition:start_time', 'study:start_time']

       # First try to get date and time together from standard MINC fields
       header = self.header
       for field in field_datetime:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})\s+(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})\.?(?P<microsecond>[0-9]+)?', out).groupdict()
            except Exception as e:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            return datetime(**m) 

       # If failed, first get date time from dicom fields
       date = None
       for field in field_date:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})', out).groupdict()
            except Exception as e:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            date = datetime(**m)
            break

       # Then get scan time
       for field in field_time:
            try:
                out = mincinfo(header, field)
                m = re.search('(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})\.?(?P<microsecond>[0-9]+)?', out).groupdict()
            except Exception as e:
                continue
            for k in m.keys():
                if m[k] is None:
                    del m[k]
                else:
                    m[k] = int(m[k])
            date.replace(**m)
            break
       if date:
           return date
       raise Exception('Could not extract acquisition date from {0}'.format(self.name))

    def compare(self, image,rtol=1e-05, atol=1e-08):
        return np.allclose(self.data, image.data, rtol=rtol, atol=atol)

    def toLabel(self):
        """Convert a copy of this instance as a Label instance.
        :rtype: a minc.Label instance.
        """
        return Label(data=self.data.astype(np.int32))
                
    def mutualInformation(self, refImage, mask = None, normalize = False):
        """Compute mutual information between two images
        
        :params Image refImage: Image with which mutual information is computed
        :params Label mask: If provided, only data within the mask are used
        :params bool normalize: If true, normalized positive values will be used 
        
        :returns: Will return mutual information value.
        """
        if mask:
            selfArray = self.data * mask.data
            refArray = refImage.data * mask.data    
        else:
            selfArray = self.data 
            refArray = refImage.data 
        
        if normalize:
            if len(selfArray.shape) == 3:
                zDim = selfArray.shape[0]
        
                for z in range(zDim):
                    maxVal = np.amax(selfArray[z,:,:])
                    if maxVal != 0:
                        selfArray[z,:,:] = (1000*selfArray[z,:,:])/maxVal
                        selfArray[z,:,:][np.where(selfArray[z,:,:]<0)] = 0 
                        
                    maxVal = np.amax(refArray[z,:,:])
                    if maxVal != 0:
                        refArray[z,:,:] = (1000*refArray[z,:,:])/maxVal
                        refArray[z,:,:][np.where(refArray[z,:,:]<0)] = 0 
                            
            elif len(selfArray.shape) == 2:
                maxVal = np.amax(selfArray)
                if maxVal != 0:
                    selfArray = (1000*selfArray)/maxVal
                    selfArray[np.where(selfArray<0)] = 0 
                    
                maxVal = np.amax(refArray)
                if maxVal != 0:
                    refArray = (1000*refArray)/maxVal
                    refArray[np.where(refArray<0)] = 0 
        
            vRange = [[0,1000],[0,1000]]
                
            if np.amax(selfArray) > 1001 or np.amax(refArray) > 1001:
                system.log_Debug('Normalization has not been done properly!!!')
                raise Exception()
        else:
            maxValue = max(np.amax(selfArray), np.amax(refArray))
            minValue = min(np.amin(selfArray), np.amin(refArray))
            vRange = [[minValue, maxValue],[minValue, maxValue]]
                      
        selfArray = selfArray.flatten()
        refArray = refArray.flatten()                  
        bins = 500
        
        miValue = mi(selfArray, refArray, bins, vRange)
    
        return miValue
            
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
