#!/usr/bin/env python

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
from glob import glob
import tempfile
import string
import shutil
import sys

import pyezminc


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
        cosines = tuple(self._get_direction_cosines()[d] for d in dimensions)
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

    def compare(self, image):
        return np.allclose(self.data, image.data)

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
            
class Label(Image):

    def __init__(self, fname=None, data=None, autoload=True, dtype=np.int32, *args, **kwargs):
        self.include_not_label = False
        super(Label, self).__init__(fname=fname, data=data, autoload=autoload, dtype=dtype, *args, **kwargs)

    def zeros(self, shape):
        self.data = np.zeros(shape, dtype=self.dtype)        

    def regions_id(self):
        '''Returns a list of valid regions ID'''
        key = list(np.unique(self.data))
        if self.include_not_label is False:
            if key[0] == 0:
                del key[0]
        return key

    def nb_regions(self):
        '''Returns the number of regions'''
        return len(self.regions_id())

    def regions_indices(self):
        '''Returns a dictionary with the coordinates for
        each regions'''
        return dict((i, np.where(self.data == i)) for i in self.regions_id())

    def values(self, idx):
        '''Given a dictionary of coordinates, it returns
        a dictionary with associated values of each voxel'''
        return dict((r, self.data[idx[r]]) for r in idx.iterkeys())

    def set_values(self, idx, order=None):
        '''Given a coordinates dictionary, it writes the value
        for each region'''
        if order is None:
            for k, v in idx.iteritems():
                self.data[v] = k
        else:
            for k in order:
                self.data[idx[k]] = k

    def regions_nbvoxels(self):
        '''Returns the number of voxels for each region'''

        bins = np.bincount(np.ravel(self.data))
        return dict((i, bins[i]) for i in self.regions_id())

    def regions_volume(self):
        '''Returns the volume for each region'''
        siz = self.regions_nbvoxels()
        return dict((i, self.volume(siz[i])) for i in siz.iterkeys())


    def split_regions(self, binary=False):
        '''Split regions id into separate Label instance'''
        indices = self.regions_indices()
        split = dict()
        for i, idx in indices.iteritems():
            split[i] = Label(data=np.zeros(self.data.shape, dtype=self.dtype))
            if binary:
                split[i].data[idx] = 1
            else:
                split[i].data[idx] = i
        return split
    
    def find_regions_with_labels(self, label_dic):
        ''' Return a binary labeled volume.
            Voxels whose original label is listed in label_dic will be set to 1 
            in the output volume.
    
            Keyword arguments:
            label_dic -- Dictionary of labels which will be used as reference.   
        '''
        indices = self.regions_indices()
        
        outLabel = Label(data=np.zeros(self.data.shape, dtype=self.dtype))
        
        for i, idx in indices.iteritems():
            if i in label_dic.values():
                outLabel.data[idx] = 1;
        
        return outLabel                

    def intersection(self, reference):
        '''Returns the indices of intersection with
        a reference label.'''
        assert isinstance(reference, Label)

        idx_s = self.regions_indices()
        idx_r = reference.regions_indices()
        stats = dict()
        for k_s,v_s in idx_s.iteritems():
            tr_s = set(tuple(i) for i in np.transpose(v_s))
            stats[k_s] = dict()
            for k_r,v_r in idx_r.iteritems():
                tr_r = set(tuple(i) for i in np.transpose(v_r))
                inter = tr_s.intersection(tr_r)
                if len(inter):
                    stats[k_s][k_r] = inter
        return stats

    def error_type(self, reference, tp_min_size=1):
        '''Returns a dictionary with tagged error type
        for each regions from both reference and this label.'''
        inter = self.intersection(reference)
        stats = {'reference': dict(), 'self': dict()}
        ref_region_not_matched = set()

        for k_s,v_s in inter.iteritems():
            # Tuple reference labelname and size of intersection
            len_inter = [(k, len(v)) for k,v in v_s.iteritems()]
            # Filter regions which satisfy TRUE POSITIVe criteria
            bool_tp = np.array([i[1] for i in len_inter]) >= tp_min_size

            # The region is a TP, find matches in the reference
            if np.any( bool_tp ):
                if not isinstance(bool_tp, np.ndarray):
                    bool_tp = np.array([bool_tp])
                for i, v in enumerate(bool_tp):
                    name = len_inter[i][0]
                    if not v:
                        continue
                    if stats['self'].has_key(k_s):
                        stats['self'][k_s]['regions'].append(name)
                    else:
                        stats['self'][k_s] = {'type': 'TP', 'regions': [name]}
                    # Remove ref region as a potention FN
                    if name in ref_region_not_matched:
                        ref_region_not_matched.remove(name)
            # The region is a FP, add all regions from ref as potential FN
            else:
                ref_region_not_matched.update(v_s.keys())
                stats['self'][k_s] = {'type': 'FP'}

        # Label all remaining regions from ref as FN
        for i in ref_region_not_matched:
            stats['reference'][i] = {'type': 'FN'}
        
        for i in reference.regions_id():
            tpFlag = False
            for v in stats['self'].itervalues():
                if v['type'] == 'TP':
                    if i in v['regions']:
                        tpFlag = True
                        break
            if not tpFlag:
                stats['reference'][i] = {'type': 'FN'}

        return stats

    def error_type_voxelWise(self, reference):
        '''Returns a dictionary with tagged error type
        for each regions from both reference and this label.'''
        
        self.data = np.where(self.data>0, 1, 0)
        reference.data = np.where(reference.data>0, 1, 0)
        
        interSet = self.intersection(reference).get(1, dict()).get(1, set())
        interArray = np.asarray(list(interSet))
        stats = {'reference': dict(), 'self': dict()}

        sIdx = np.where(self.data>0)
        rIdx = np.where(reference.data>0)
        
        sIdxArray = np.transpose(np.asarray(sIdx))
        rIdxArray = np.transpose(np.asarray(rIdx))
            
        for s in sIdxArray:
            if interArray.any():
                if np.equal(s, interArray).all(axis=1).any():
                    stats['self'][str(s)] = {'type': 'TP', 'regions': 1}
                else:
                    stats['self'][str(s)] = {'type': 'FP'}
            else:
                stats['self'][str(s)] = {'type': 'FP'}

        for r in rIdxArray:
            if interArray.any():
                if not np.equal(r, interArray).all(axis=1).any():
                    stats['reference'][str(r)] = {'type': 'FN'}
            else:
                stats['reference'][str(r)] = {'type': 'FN'}

        return stats

    def count_error_type(self, reference, tp_min_size=1):
        '''Returns how many FN, FP and TP there are
        when compared to reference'''
        stats = self.error_type(reference, tp_min_size)
        summary = {'FN': len(stats['reference']),
                   'FP': len([i for i in stats['self'].itervalues() if i['type']=='FP']),
                   'TP': len([i for i in stats['self'].itervalues() if i['type']=='TP'])
                   }
        return summary


    def get_TP(self, reference, tp_min_size=1, voxelWise=False):
        '''Get a dictionary of all TP regions and their matching region'''
        if voxelWise:
            stats = self.error_type_voxelWise(reference)
        else:
            stats = self.error_type(reference, tp_min_size)
        
        return dict((k, v['regions'])
                    for k,v in stats['self'].iteritems() if v['type']=='TP')

    def get_FP(self, reference, tp_min_size=1, voxelWise=False):
        '''Get a dictionary of all FP regions and their matching region'''
        if voxelWise:
            stats = self.error_type_voxelWise(reference)
        else:
            stats = self.error_type(reference, tp_min_size)
            
        return dict((k, None)
                    for k,v in stats['self'].iteritems() if v['type']=='FP')

    def get_FN(self, reference, tp_min_size=1, voxelWise=False):
        '''Get a dictionary of all FN regions and their matching region'''
        if voxelWise:
            stats = self.error_type_voxelWise(reference)
        else:
            stats = self.error_type(reference, tp_min_size)
            
        return dict((k, None)
                    for k,v in stats['reference'].iteritems() if v['type']=='FN')

    def get_TN(self, reference, tp_min_size=1, voxelWise=False):
        '''Get a dictionary of all TN regions and their matching region'''
        if voxelWise:
            stats = self.error_type_voxelWise(reference)
        else:
            stats = self.error_type(reference, tp_min_size)
            
        return dict((k, v['regions'])
                    for k,v in stats['self'].iteritems() if v['type']=='TN')

    def compare_volume_sympercents(self, label):
        '''Compare with another label. Return a tuple of regionwise and total percentage'''
        tp1 = self.get_TP(label)
        tp2 = label.get_TP(self)
        vol1 = self.regions_nbvoxels()
        vol2 = label.regions_nbvoxels()

        sympct_regionwise = {}
        for r1, lst in tp1.iteritems():
            r2 = lst[0]
            # Several regions overlapping
            if len(lst) > 1 or len(tp2[r2]) > 1:
                continue
            vol1f = vol1[r1]
            vol2f = vol2[r2]
            val = 200*(vol1f-vol2f) / (vol1f+vol2f)
            sympct_regionwise[r1] = val

        return sympct_regionwise

    def compare_total_volume_sympercents(self, label):
        vol1 = sum(self.regions_nbvoxels().values())
        vol2 = sum(label.regions_nbvoxels().values())
        try:
            sympct_total = 200*(vol1-vol2) / (vol1+vol2)
        except ZeroDivisionError:
            sympct_total = np.nan
        return sympct_total

    def dilation(self):
        '''Dilate every region by an 8-connected
        in-plane structure'''
        struct = ndimage.generate_binary_structure(3,3)
        struct[0] = False
        struct[2] = False
        l = copy.copy(self)
        l.data = ndimage.grey_dilation(self.data, size=(3,3,3), footprint=struct).astype(self.data.dtype)
        return l
    
    def binary_dilation(self, struct = None, mask = None, iterations = 1):
        """ Perform binary dilation.
            The default behavior is to dilate in-pane using a structure with level-1 connectivity   
    
        :param ndarray struct: Binary structure to use for dilation
        :param ndarray mask: Dilation can only happen within the mask
        :param int iterations: Dilation is repeated iterations times
        :returns: The new dilated label file
        :rtype: Label object
        """
            
        if struct == None:
            struct = ndimage.generate_binary_structure(3,1)
            struct[0] = False
            struct[2] = False
            
        dilatedLabel = copy.deepcopy(self)            
        dilatedLabel.data = ndimage.binary_dilation(self.data, 
                                                    structure = struct, 
                                                    iterations = iterations, 
                                                    mask = mask).astype(self.data.dtype)
        return dilatedLabel
    
    def binary_erosion(self, struct = None, mask = None, iterations = 1):
        """ Perform binary erosion.
            The default behavior is to erode in-pane using a structure with level-1 connectivity   
    
        :param ndarray struct: Binary structure to use for erosion
        :param ndarray mask: Erosion can only happen within the mask
        :param int iterations: Erosion is repeated iterations times
        :returns: The new eroded label file
        :rtype: Label object
        """
        
        if struct == None:
            struct = ndimage.generate_binary_structure(3,1)
            struct[0] = False
            struct[2] = False
            
        erodedLabel = copy.deepcopy(self)            
        erodedLabel.data = ndimage.binary_erosion(self.data, 
                                                  structure = struct, 
                                                  iterations = iterations, 
                                                  mask = mask).astype(self.data.dtype)
        return erodedLabel

    def remove_Small_Regions(self, minNumVoxels = 3):
        """ Remove any region smaller than minNumVoxels and reassign labels to
            the remaining regions
            :param int minNumVoxels: The minimum number of voxels required for each region
        """
        
        regionsSize = self.regions_nbvoxels()
        regionsIndices = self.regions_indices()
        
        for i in self.regions_id():
            if regionsSize[i] < minNumVoxels:
                self.data[regionsIndices[i]] = 0
        
        '''Reassign labels to differenr connected regions'''
        struct = ndimage.generate_binary_structure(3, 2)
        connectedCompsImage = ndimage.label(self.data, structure = struct)[0]
        self.data = connectedCompsImage        

    def toMask(self):
        """Convert a copy of this instance as a Mask instance.
        :rtype: a minc.Mask instance.
        """
        return Mask(data=self.data, mnc2np=True)
        

class Mask(Image):
    def __init__(self, fname=None, data=None, autoload=True, dtype=np.int32, mnc2np=True, *args, **kwargs):
        self.mnc2np = mnc2np
        super(Mask, self).__init__(fname=fname, data=data, autoload=autoload, dtype=dtype, *args, **kwargs)
        if mnc2np and data!=None and not fname:
            self.data = self._minc_to_numpy(self.data)
    
    def _minc_to_numpy(self, data):
        return np.logical_not(data.astype(np.bool)).astype(np.dtype)
    
    def _numpy_to_minc(self, data):
        return np.logical_not(data.astype(np.bool)).astype(self.dtype)
        
    def _load(self, name, *args, **kwargs):
        '''Load a file'''
        super(Mask, self)._load(name, *args, **kwargs)
        if self.mnc2np:
            self.data = self._minc_to_numpy(self.data)
        
    def _save(self, name=None, imitate=None, *args, **kwargs):
        old_data = self.data
        try:
            self.data = self._numpy_to_minc(self.data)
            super(Mask, self)._save(name=name, imitate=imitate, *args, **kwargs)
        finally:
            self.data = old_data

def main(argv=None):
    DATA_PATH = os.path.join(os.path.split(os.path.abspath( __file__ ))[0], 'test')

    label_name = os.path.join(DATA_PATH, 'trial_site00_subject_00_screening_gvf_ISPC-stx152lsq6.mnc.gz')
    l = minc.Label(label_name)
    
    img_name = os.path.join(DATA_PATH, 'trial_site01_subject_00_screening_t2w.mnc.gz')
    img = minc.Image(img_name)

    print('All appears OK')

if __name__ == '__main__':
    main()

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
