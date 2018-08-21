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

import unittest
import doctest
import string
import os
#import xmlrunner
import numpy as np
import numpy.ma as ma
import subprocess as sp
import tempfile
import shlex

import minc

DATA_PATH =    'test/data'
ANAT_FILE =    'mri_t1.mnc'
ANAT_FILE_T2 = 'mri_t2.mnc'
MASK_FILE =    'mask.mnc'
LABEL_FILE=    'atlas_csf.mnc'
NAN_FILE  =    'nan.mnc'

def check_call_out(command, cwd=None, autosplit=True, shell=False, verbose=False, logFile=None, stderr=sp.STDOUT, print_exception=True, *args, **kwargs):
    """ Execute a command, grab its output and check its return code.

    :param command: command and its options.
    :type command: string or list of strings
    :param string cwd: the path where to run the command.
    :param bool autosplit: automatically split the command into a list when the command and options are given as a string (as opposed to a list).
    :param bool shell: run the command with the shell. Not recommanded unless you need the glob pattern.
    :param bool verbose: print the command with options that is being run.
    :param file logFile: append the output to an exisiting file instead of creating a temporary one.
    :param file stderr: how to handle the standard error stream. By default, merge it with stdout.
    :param bool print_exception: print the output of command when there is an exception.
    :returns: the output of the command.
    :rtype: string
    :raises: subprocess.CalledProcessError if the command does not return 0.
    """
    if autosplit and not shell and isinstance(command, basestring):
        command = shlex.split(command)

    if verbose:
        if isinstance(command, basestring):
            print command
        else:
            print string.join(command, sep=' ')

    if not logFile:
        fpipe = tempfile.NamedTemporaryFile()
    else:
        fpipe = logFile
    
    try:
        # Quick fix for strange bug with shell, saying no such file
        if shell:
            p = sp.Popen(command, cwd=cwd, stdout=sp.PIPE, stderr=stderr, shell=True)
            buf = p.communicate()[0]
            if p.returncode:
                raise sp.CalledProcessError(p.returncode, command)
        else:
            if not logFile:
                sp.check_call(command, cwd=cwd, stdout=fpipe.fileno(), stderr=stderr)
                fpipe.seek(0)
                buf = fpipe.read()
            else:
                sp.check_call(command, cwd=cwd, stdout=logFile, stderr=stderr)
                logFile.seek(0)
                buf = logFile.read()
    except OSError as e:
        if isinstance(command, basestring):
            command = shlex.split(command)
        e.filename = command[0]
        raise e
    except sp.CalledProcessError as e:
        if not logFile:
            fpipe.seek(0)
            buf = fpipe.read()

            if print_exception:
                if not verbose:
                    print command
                print buf
            e.output = buf
        raise e    
    finally:
        if not logFile:
            fpipe.close()

    return buf


def create_tmp_filename (suffix='.mnc.gz', prefix='tmp_', remove=True):
    '''Create a unique temporary file in the system and returns its name.

    :param string suffix: suffix to customize the end of the temporary filename.
    :param string prefix: prefix to customize the beginning of the temporary filename.
    :param bool remove: remove the temporary file removed. It may be useful in somes cases, but it creates a racing condition in which two temporary file can have the same name.
    :returns: the temporary filename. DO NOT forget to delete the file once you
        do not need it anymore! (os.remove)
    :rtype: string

    '''
    tmp_fd, tmp_filename = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(tmp_fd)
    if remove:
        os.remove(tmp_filename)
    return tmp_filename


class TestIterator(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, ANAT_FILE)
        self.tmp = create_tmp_filename(prefix='iter', suffix='.mnc', remove=False)

    def tearDown(self):
        pass
  
    def testInputSum(self):
        it=minc.input_iterator_real(self.fname)
        sum=0.0
        cnt=0
        for i in it:
            sum+=i
            cnt+=1

        self.assertEqual(cnt, 308800)
        self.assertAlmostEqual(sum, 17708200.75, places=2)

    def testOutput(self):
        it_in=minc.input_iterator_real(self.fname)
        it_out=minc.output_iterator_real(self.tmp, reference_file=self.fname)
        for i in it_in:
            it_out.value(i)
            it_out.next()
        # compare now
        del it_out
        del it_in

        _ref=minc.Image(self.fname).data
        _out = minc.Image(self.tmp).data

        self.assertTrue(np.allclose(_ref,_out))


class TestLabel(unittest.TestCase):

    def setUp(self):

        gvf = LABEL_FILE
        self.fname = os.path.join(DATA_PATH, gvf)
        self.img = minc.Label(self.fname)
        self.tmp = create_tmp_filename(prefix='atlas_csf', suffix='.mnc', remove=False)

    def tearDown(self):
        del self.img
        if os.path.isfile(self.tmp):
            os.remove(self.tmp)

    def testSaveDtype(self):
        ''' Make sure that saving an image with wrong dtype raises an exception. '''
        self.img.dtype = np.float64
        self.assertNotEqual(self.img.dtype, self.img.data.dtype)
        with self.assertRaises(Exception):
            self.img.save(self.tmp)

    def testSaveDtype2(self):
        ''' Make sure that saving an image with wrong dtype raises an exception. '''
        self.img.data.dtype = np.float64
        self.assertNotEqual(self.img.dtype, self.img.data.dtype)
        with self.assertRaises(Exception):
            self.img.save(self.tmp)

    def testRegionsId(self):
        self.assertEqual(self.img.regions_id(), [3, 9, 232, 233, 255])
        
    def testRegionsIndices(self):
        nb = len(self.img.regions_indices()[3][0])
        self.assertEqual(nb, 8219)

    def testSplitRegions(self):
        split = self.img.split_regions()
        nb_regions = dict((k, r.nb_regions()) for k,r in split.items())
        self.assertEqual(nb_regions, {3:1, 9:1, 232:1 , 233:1 ,255: 1})
            
    def testNbRegions(self):
        self.assertEqual(self.img.nb_regions(), 5)

    def testVolume(self):
        volume_py = [self.img.regions_volume()[k] for k in sorted(self.img.regions_volume().keys())]
        volume_c =  [float( i.split(' ')[2] ) for i in check_call_out('print_all_labels {}'.format(self.fname)).rstrip("\n").split("\n")]
        self.assertEqual(volume_py, volume_c)

    #@unittest.skipIf(not os.path.isdir('/trials/quarantine'), '/trials/quarantine directory does not exist')
    #def testDilation(self):
        #tmp = {}
        #tmp_orig_root = create_tmp_filename(suffix='', remove=True)
        #tmp['orig'] = tmp_orig_root + '_d.mnc'
        #tmp['cmp'] = create_tmp_filename(suffix='', remove=True)

        ## Dilate the label with python
        #l = self.img.dilation()
        #l.save(self.tmp)

        ## Dilate the label with minc-ed
        #check_call_out('minc-ed {input} {outroot} d'.format(input=self.fname, outroot=tmp_orig_root), cwd='/tmp')

        ## Compare labels
        #check_call_out('mincmath -clobber -float -sub {python} {original} {output}'.format(python=self.tmp, original=tmp['orig'], output=tmp['cmp']))
        #out = check_call_out('mincstats -sum {cmp}'.format(cmp=tmp['cmp']))
        #res = float(re.sub('Sum:(.*)',r'\1', out))
        #self.assertAlmostEqual(res, 0)

        #for f in tmp.itervalues():
            #if os.path.isfile(f):
                #os.remove(f)

class TestImage(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, ANAT_FILE)
        self.img = minc.Image(self.fname)
        self.fname_nan = os.path.join(DATA_PATH, ANAT_FILE)
        self.tmp = create_tmp_filename(prefix='mni_icbm152_', suffix='.mnc', remove=False)

    def tearDown(self):
        if os.path.isfile(self.tmp):
            os.remove(self.tmp)
        del self.img

    def testLoad(self):
        img = minc.Image()
        img.load(self.fname)
        self.assertTrue(isinstance(img.data, np.ndarray))

    def testShape(self):
        # numpy array is slowest varying first
        self.assertEqual(self.img.data.shape, (193, 40, 40))

    def testDim(self):
        # EZminc dimensions - X,Y,Z always
        self.assertEqual(self.img.dim(), [40, 40, 193])

    def testSpacing(self):
        # EZminc notation - X,Y,Z order
        self.assertEqual(self.img.spacing(), [1, 1, 1])

    def testVolume(self):
        self.assertAlmostEqual(self.img.volume(1), 1.0)

    def testHistory(self):
        img = minc.Image(self.fname)
        history = [
            'Thu Jul 30 14:23:47 2009>>> mincaverage -short mni_icbm152_t1_tal_nlin_sym_09c_.mnc mni_icbm152_t1_tal_nlin_sym_09c_flip_.mnc final/mni_icbm152_t1_tal_nlin_sym_09c.mnc',
            'Thu Apr  1 15:39:40 2010>>> mincconvert ./mni_icbm152_t1_tal_nlin_sym_09c.mnc ./mni_icbm152_t1_tal_nlin_sym_09c.mnc.minc1',
            'Mon Aug 13 12:55:20 2018>>> mincresample -nearest -like test/data/atlas_csf.mnc /data/vfonov/models/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc test/data/mri_t1.mnc']
        self.assertEqual(img.history, history)
    
    def testDirectionCosines(self):
        cosines = {'xspace': (1.0, 0.0, 0.0),
                   'yspace': (0.0, 1.0, 0.0),
                   'zspace': (0.0, 0.0, 1.0)}
        self.assertEqual(self.img.direction_cosines, cosines)

    def testNumpyType(self):
        self.assertTrue(self.img.dtype == np.float64)
        self.assertTrue(self.img.data.dtype == np.float64)
        
    def testStart(self):
        start = [-26, -72, -78 ]
        self.assertEqual(self.img.start(), start)

    def testVoxelToWorld(self):
        coords = [-16, -52, -48]
        test_coords = self.img.voxel_to_world((10,20,30))

        self.assertEqual(coords, test_coords)

    def testSave(self):
        self.img.save(self.tmp)
        self.assertTrue(os.path.isfile(self.tmp))

    def testSaveDtype(self):
        ''' Make sure that saving an image with wrong dtype raises an exception. '''
        self.img.dtype = np.int32
        self.assertNotEqual(self.img.dtype, self.img.data.dtype)
        with self.assertRaises(Exception):
            self.img.save(self.tmp)

    def testSaveDtype2(self):
        ''' Make sure that saving an image with wrong dtype raises an exception. '''
        self.img.data.dtype = np.int32
        self.assertNotEqual(self.img.dtype, self.img.data.dtype)
        with self.assertRaises(Exception):
            self.img.save(self.tmp)

    def testMax(self):
        self.assertAlmostEqual(self.img.data.max(), 93.4918238)

    def testMin(self):
        self.assertAlmostEqual(self.img.data.min(), 3.7532188)

    def testMean(self):
        self.assertAlmostEqual(np.mean(self.img.data), 57.34520967)

    def testMedian(self):
        self.assertAlmostEqual( np.median(self.img.data), 63.23888279, places=3 ) # median in minstats is different

    def testVariance(self):
        self.assertAlmostEqual(np.var(self.img.data, ddof=1), 558.4608597)

    def testSize(self):
        self.assertEqual(np.size(self.img.data), 40*40*193)

    def testVolume(self):
        self.assertAlmostEqual(self.img.volume(np.size(self.img.data)), 308800)


    def testMincType(self):
        for minctype in ('byte', 'short', 'int', 'float', 'double'):
            check_call_out(['mincreshape', '-'+minctype, '-clobber', self.fname, self.tmp])
            self.img = minc.Image(self.tmp)
            self.assertAlmostEqual(np.median(self.img.data), 63.245, places=2 )

    def testLoadNan(self):
        with self.assertRaises(Exception):
            img = minc.Image(os.path.join(DATA_PATH, NAN_FILE))
            # TODO: make a volume with NaN

    def testLoadHeader(self):
        self.assertTrue(self.img.header)
        # TODO: make a test for header


class TestMaskedImage(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, ANAT_FILE_T2)
        self.fmask = os.path.join(DATA_PATH, MASK_FILE)
        self.masked_data = ma.masked_array(minc.Image(self.fname).data, mask=minc.Mask(self.fmask).data)
        self.tmp = create_tmp_filename(prefix='mni_icbm152_t2_', suffix='.mnc', remove=True)

    def tearDown(self):
        if os.path.isfile(self.tmp):
            os.remove(self.tmp)
        del self.masked_data

    def testMax(self):
        self.assertAlmostEqual(self.masked_data.max(), 103.3849068)

    def testMin(self):
        self.assertAlmostEqual(self.masked_data.min(), 8.290930849)

    def testMean(self):
        self.assertAlmostEqual(ma.mean(self.masked_data), 55.43792888 )

    def testMedian(self):
        self.assertAlmostEqual(ma.median(self.masked_data), 51.51, places=1 ) # median in numpy and mincstats are different

    def testVariance(self):
        self.assertAlmostEqual(ma.var(self.masked_data, ddof=1), 206.5207728)

    def testCount(self):
        self.assertEqual(ma.count(self.masked_data), 217365)

    def testSum(self):
        self.assertAlmostEqual(ma.sum(self.masked_data), 12050265.41,places=2)


class TestXFM(unittest.TestCase):
    def setUp(self):
        self.tmp = create_tmp_filename(prefix='transform', suffix='.xfm', remove=True)

    def tearDown(self):
        if os.path.isfile(self.tmp):
            os.remove(self.tmp)

    def testXfmRead(self):
        # generate xfm file first
        check_call_out(["param2xfm","-rotations","30","0","0",self.tmp,"-clobber"])
        xfm=minc.read_xfm(self.tmp)
        #print(xfm)
        reference_matrix=np.array([
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.866025388240814, -0.5, 0.0],
             [0.0, 0.5, 0.866025388240814, 0.0],
             [0.0, 0.0, 0.0, 1.0 ]
            ])
        self.assertTrue(len(xfm) == 1)
        self.assertTrue(xfm[0].lin)
        self.assertTrue(np.allclose(reference_matrix, xfm[0].trans))

    def testXfmWrite(self):
        # generate xfm file first
        reference_matrix=np.array([
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.866025388240814, -0.5, 0.0],
             [0.0, 0.5, 0.866025388240814, 0.0],
             [0.0, 0.0, 0.0, 1.0 ]
            ])
        reference_transform=[minc.xfm_entry(True,False,reference_matrix)]
        minc.write_xfm(self.tmp, reference_transform, 'Reference')

        with open(self.tmp,'r') as f:
            ln=f.read()

        ref_ln="""MNI Transform File
%Reference

Transform_Type = Linear;
Linear_Transform =
 1 0 0 0
 0 0.866025388240814 -0.5 0
 0 0.5 0.866025388240814 0;
"""
        self.assertEqual(ln,ref_ln)

    def testXfmToParam(self) :
        def compare_parameters(par,ref):
            if not np.allclose(ref.rotations, par.rotations):
                print("rotations mismatch:",ref.rotations, par.rotations)
                return False
            if not np.allclose(ref.translations, par.translations):
                print("translations mismatch:",ref.translations, par.translations)
                return False
            if not np.allclose(ref.scales, par.scales):
                print("scales mismatch:",ref.translations, par.translations)
                return False
            if not np.allclose(ref.translations, par.translations):
                print("translations mismatch:",ref.translations, par.translations)
                return False
            if not np.allclose(ref.shears, par.shears):
                print("shears mismatch:",ref.shears, par.shears)
                return False
            return True

        for i in range(3):
            for r in [-10, -10,  10, 20]:
                ref = minc.xfm_identity_transform_par()
                ref.rotations[i] = r
                cmd=["param2xfm", "-rotations"]+[str(j) for j in ref.rotations]+[self.tmp, "-clobber"]

                check_call_out(cmd)
                par = minc.xfm_to_param(minc.read_xfm(self.tmp))

                self.assertTrue(compare_parameters(ref,par),"Error in rotations i={} r={}".format(i,r))

        for i in range(3):
            for r in [-10, -10,  10, 20]:
                ref = minc.xfm_identity_transform_par()
                ref.translations[i] = r
                cmd=["param2xfm", "-translation"]+[str(j) for j in ref.translations]+[self.tmp, "-clobber"]

                check_call_out(cmd)
                par = minc.xfm_to_param(minc.read_xfm(self.tmp))

                self.assertTrue(compare_parameters(ref,par),"Error in translations i={} r={}".format(i,r))

        for i in range(3):
            for r in [0.9, 1.1,  1.2, 1.3]:
                ref = minc.xfm_identity_transform_par()
                ref.scales[i] = r
                cmd=["param2xfm", "-scales"]+[str(j) for j in ref.scales]+[self.tmp, "-clobber"]

                check_call_out(cmd)
                par = minc.xfm_to_param(minc.read_xfm(self.tmp))

                self.assertTrue(compare_parameters(ref,par),"Error in scales i={} r={}".format(i,r))

    def testParamToXFM(self) :
        def compare_matrices(par, ref):
            if not np.allclose(par,ref):
                print("Matrix mismatch:", ref, par)
                return False
            return True

        ref = minc.xfm_identity_transform_par()
        par_mat = minc.param_to_xfm(ref)

        for i in range(3):
            for r in [-10, -10,  10, 20]:
                ref = minc.xfm_identity_transform_par()
                ref.translations[i] = r
                cmd=["param2xfm", "-translation"] + [str(j) for j in ref.translations] + [self.tmp, "-clobber"]
                check_call_out(cmd)
                ref_mat = minc.read_xfm(self.tmp)
                par_mat = minc.param_to_xfm(ref)
                self.assertTrue(par_mat.lin)
                self.assertFalse(par_mat.inv)
                self.assertTrue( compare_matrices(par_mat.trans, ref_mat[0].trans),"Error in translations i={} r={}".format(i,r))

        for i in range(3):
            for r in [-10, -10,  10, 20]:
                ref = minc.xfm_identity_transform_par()
                ref.rotations[i] = r
                cmd=["param2xfm", "-rotations"]+[str(j) for j in ref.rotations]+[self.tmp, "-clobber"]
                check_call_out(cmd)
                ref_mat=minc.read_xfm(self.tmp)
                par_mat=minc.param_to_xfm(ref)
                self.assertTrue(par_mat.lin)
                self.assertFalse(par_mat.inv)
                self.assertTrue( compare_matrices(par_mat.trans, ref_mat[0].trans),"Error in rotations i={} r={}".format(i,r))


        for i in range(3):
            for r in [0.9, 1.1,  1.2, 1.3]:
                ref = minc.xfm_identity_transform_par()
                ref.scales[i] = r
                cmd=["param2xfm", "-scales"]+[str(j) for j in ref.scales]+[self.tmp, "-clobber"]
                check_call_out(cmd)
                ref_mat=minc.read_xfm(self.tmp)
                par_mat=minc.param_to_xfm(ref)
                self.assertTrue(par_mat.lin)
                self.assertFalse(par_mat.inv)
                self.assertTrue( compare_matrices(par_mat.trans,ref_mat[0].trans),"Error in scales i={} r={}".format(i,r))

if __name__ == "__main__":
    unittest.main()

