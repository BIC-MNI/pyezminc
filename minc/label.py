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

import numpy as np
from . import Image

FIRST_LABEL_ID = 1

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
        

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
