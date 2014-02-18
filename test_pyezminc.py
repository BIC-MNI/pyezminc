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

try:
    #python2.6
    import unittest2 as unittest
except ImportError:
    #python2.7
    import unittest
import doctest
import string
import os
import re
#import xmlrunner
import argparse
import sys
import numpy as np
import numpy.ma as ma
import sys
import subprocess as sp
import tempfile
import shlex

import minc


DATA_PATH = '/opt/minc/share/icbm152_model_09c'

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
        self.fname = os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc')
        
    def tearDown(self):
        pass
  
    def testSum(self):
        it=input_iterator_real(self.fname)
        sum=0.0
        for i in it:
            sum+=i
        print "sum={}".format(sum)
        self.assertEqual(sum,)

class TestLabel(unittest.TestCase):

    def setUp(self):

        gvf = 'mni_icbm152_t1_tal_nlin_sym_09c_atlas/atlas_csf.mnc'
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
        nb = len(self.img.regions_indices()[2][0])
        self.assertEqual(nb, 31)

    def testSplitRegions(self):
        split = self.img.split_regions()
        nb_regions = dict((k, r.nb_regions()) for k,r in split.iteritems())
        self.assertEqual(nb_regions, {1:1, 2:1})
            
    def testNbRegions(self):
        self.assertEqual(self.img.nb_regions(), 2)

    #@unittest.skipIf(not os.path.isdir('/trials/quarantine'), '/trials/quarantine directory does not exist')
    #def testVolume(self):
        #volume_py = [v for v in self.img.regions_volume().itervalues()]
        #volume_c = float(check_call_out('label_volume.pl {0}'.format(self.fname)))
        #self.assertAlmostEqual(sum(volume_py), volume_c)

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
        self.fname = os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc')
        self.img = minc.Image(self.fname)
        self.fname_nan = os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc')
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
        self.assertEqual(self.img.data.shape, (193,229,193))

    def testDim(self):
        self.assertEqual(self.img.dim(), [193,229,193])

    def testSpacing(self):
        self.assertEqual(self.img.spacing(), [1,1,1])

    def testVolume(self):
        self.assertAlmostEqual(self.img.volume(1), 1.0)

    def testHistory(self):
        img = minc.Image(self.fname)
        history = ['Thu Jul 30 14:23:47 2009>>> mincaverage -short mni_icbm152_t1_tal_nlin_sym_09c_.mnc mni_icbm152_t1_tal_nlin_sym_09c_flip_.mnc final/mni_icbm152_t1_tal_nlin_sym_09c.mnc']
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
        start = [-78, -132, -96 ]
        self.assertEqual(self.img.start(), start)

    def testVoxelToWorld(self):
        coords = [-66, -112, -68 ]
        self.assertTrue(np.allclose(coords, self.img.voxel_to_world((10,20,30))))

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
        self.assertAlmostEqual(self.img.data.max(), 98.55860519)

    def testMin(self):
        self.assertAlmostEqual(self.img.data.min(), 0.01061781053)

    def testMean(self):
        self.assertAlmostEqual(np.mean(self.img.data), 29.61005195)

    def testMedian(self):
        self.assertAlmostEqual(np.median(self.img.data), 12.6001232)

    def testVariance(self):
        self.assertAlmostEqual(np.var(self.img.data, ddof=1), 871.0675483)

    def testSize(self):
        self.assertEqual(np.size(self.img.data), 3932160)

    def testVolume(self):
        self.assertAlmostEqual(self.img.volume(np.size(self.img.data)), 11250896.728525057)

    def testDimensions(self):
        self.assertEqual(self.img.data.shape, (60,256,256))

    def testSpacing(self):
        self.assertTrue(np.allclose(self.img.spacing(), (-0.9766, -0.9766, 3.00001)))

    def testSum(self):
        self.assertAlmostEqual(np.sum(self.img.data), 470143597.89866185)

    def testFieldStrength(self):
        self.assertEqual(self.img.field_strength(), 1.5)
        
    def testMincType(self):
        for minctype in ('byte', 'short', 'int', 'float', 'double'):
            check_call_out(['mincreshape', '-'+minctype, '-clobber', self.fname, self.tmp])
        self.img = minc.Image(self.fname)
        self.assertAlmostEqual(np.median(self.img.data), 12.6001232)

    def testLoadNan(self):
        with self.assertRaises(Exception):
            img = minc.Image(self.fname_nan)

    def testLoadHeader(self):
        self.assertTrue(self.img.header)

class TestMaskedImage(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, 'mni_icbm152_t2_tal_nlin_sym_09c.mnc')
        self.fmask = os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc')
        self.masked_data = ma.masked_array(minc.Image(self.fname).data, mask=minc.Mask(self.fmask).data)
        self.tmp = create_tmp_filename(prefix='mni_icbm152_t2_', suffix='.mnc', remove=True)

    def tearDown(self):
        if os.path.isfile(self.tmp):
            os.remove(self.tmp)
        del self.masked_data

    def testMax(self):
        self.assertAlmostEqual(self.masked_data.max(), 1139.1413919413919)

    def testMin(self):
        self.assertAlmostEqual(self.masked_data.min(), 0)

    def testMean(self):
        self.assertAlmostEqual(ma.mean(self.masked_data), 472.95778344234009)

    def testMedian(self):
        self.assertAlmostEqual(ma.median(self.masked_data), 464.10256410256409)

    def testVariance(self):
        self.assertAlmostEqual(ma.var(self.masked_data, ddof=1), 25669.716886585673)

    def testCount(self):
        self.assertEqual(ma.count(self.masked_data), 569237)

    def testSum(self):
        self.assertAlmostEqual(ma.sum(self.masked_data), 269225069.77336735)




def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run the unit tests on the pynrx module')
    parser.add_argument("--xml",
                    action="store_true",
                    dest="xml",
                    default=False,
                    help="xml report")
    parser.add_argument('unittest_args', nargs='*')

    options = parser.parse_args()
    # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1:] = options.unittest_args

    return options


def main(argv=None):
    #import sys;sys.argv = ['', 'Test.testName']

    options = parse_options()

    print 'Using DATA from {0}'.format(DATA_PATH)
    
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestIterator))
    suite.addTests(unittest.makeSuite(TestLabel))
    suite.addTests(unittest.makeSuite(TestImage))
    suite.addTests(unittest.makeSuite(TestMaskedImage))
    #suite.addTests(doctest.DocTestSuite(name))

#    if options.xml:
#        unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
#    else:
    results = unittest.TextTestRunner(verbosity = 2).run(suite)
    return not results.wasSuccessful()

if __name__ == "__main__":
    sys.exit(main())
