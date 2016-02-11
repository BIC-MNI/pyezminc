#! /usr/bin/env python

import argparse
import subprocess
import traceback
import os
import numpy as np

# needed for matrix log and exp
import scipy.linalg

# needed to read and write XFM files
import pyezminc


def do_cmd(cmds,verbose=False):
    try:
        if not verbose:
            with open(os.devnull, "w") as fnull:
                p=subprocess.Popen(cmds, stdout=fnull, stderr=subprocess.PIPE)
        else:
            p=subprocess.Popen(cmds, stderr=subprocess.PIPE)
        
        (output,output_stderr)=p.communicate()
        outvalue=p.wait()
    except OSError:
        raise Exception("ERROR: command {} Error:{}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()))
    if not outvalue == 0:
        raise Exception("ERROR: command {} failed {}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()))
    return outvalue


def xfmavg(inputs,output,verbose=False):
    # TODO: handl inversion flag correctly
    all_linear=True
    all_nonlinear=True
    input_xfms=[]
    
    for j in inputs:
        x=pyezminc.read_transform(j)
        if x[0][0] and len(x)==1 and (not x[0][1]):
            # this is a linear matrix
            input_xfms.append(x[0])
        else:
            all_linear&=False
            # strip identity matrixes
            nl=[]
            _identity=np.asmatrix(np.identity(4))
            _eps=1e-6
            for i in x:
                if i[0]:
                     if scipy.linalg.norm(_identity-i[2])>_eps: # this is non-identity matrix
                        all_nonlinear&=False
                else:
                    nl.append(i)
            if len(nl)!=1: 
                all_nonlinear&=False
            else:
                input_xfms.append(nl[0])
                
    if all_linear:
        acc=np.asmatrix(np.zeros([4,4],dtype=np.complex))
        for i in input_xfms:
            acc+=scipy.linalg.logm(i[2])
        acc/=len(input_xfms)
        out_xfm=[(True,False,scipy.linalg.expm(acc).real)]
        pyezminc.write_transform(output,out_xfm)
        
    elif all_nonlinear:
        input_grids=[]
        for i in input_xfms:
            input_grids.append(i[2])
        output_grid=output.rsplit('.xfm',1)[0]+'_grid_0.mnc'
        
        cmds=['mincaverage','-clob']
        cmds.extend(input_grids)
        cmds.append(output_grid)
        do_cmd(cmds,verbose=verbose)
        out_xfm=[(False,False,output_grid)]
        pyezminc.write_transform(output,out_xfm)
    else:
        raise Exception("Mixed XFM files provided as input")

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Perform ')

    parser.add_argument("--verbose",
                    action="store_true",
                    dest="verbose",
                    default=False,
                    help="Print verbose information" )
    
    parser.add_argument("--clobber",
                    action="store_true",
                    dest="clobber",
                    default=False,
                    help="Overwrite output" )

    parser.add_argument("inputs",
                        nargs='*',
                        help="Input xfm files")

    parser.add_argument("output",
                        help="Output xfm file")
    
    options = parser.parse_args()

    return options    


if __name__ == '__main__':
    options = parse_options()
    if options.inputs is not None and options.output is not None:
        if not options.clobber and os.path.exists(options.output):
            raise Exception("File {} exists! Run with --cloberr to overwrite".format(options.output))
        
        xfmavg(options.inputs,options.output,verbose=options.verbose)
    else:
        print("Refusing to run without input data, run --help")
        exit(1)


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
