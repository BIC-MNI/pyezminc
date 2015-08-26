#! /usr/bin/env python


import numpy as np
import scipy 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import minc
import argparse

import matplotlib.cm  as cmx
import matplotlib.colors as colors

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
    
    parser.add_argument('input',help="Run classifier on a set of given images")
    
    parser.add_argument('output',help="Output image")
    
    options = parser.parse_args()
    
    return options



if __name__ == "__main__":
    options = parse_options()
    
    _img=minc.Image(options.input)
    data_shape=_img.data.shape
    #zoom=2.0
    samples=10
    slices=[]
    
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=120)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    # show 10 coronal,axial and sagittal slices
    
    for i in range(0,data_shape[0],data_shape[0]/(samples-1)):
        slices.append( scalarMap.to_rgba(
                _img.data[i , : ,:]
            ))
        
    for i in range(0,data_shape[1],data_shape[1]/(samples-1)):
        slices.append( scalarMap.to_rgba(
                _img.data[: , i ,:]
            ))
    
    for i in range(0,data_shape[2],data_shape[2]/(samples-1)):
        slices.append( scalarMap.to_rgba(
                _img.data[: , : , i]
            ))
    
    w, h = plt.figaspect(3.0/samples)
    fig = plt.figure(figsize=(w,h))
    
    #outer_grid = gridspec.GridSpec((len(slices)+1)/2, 2, wspace=0.0, hspace=0.0)
    
    for i,j in enumerate(slices):
        ax =  plt.subplot2grid( (3, samples), (i/samples, i%samples) )
        imgplot = ax.imshow(j,origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_visible(False)
        
    plt.subplots_adjust(bottom=0.0,top=1.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)
    #fig.tight_layout()
    #plt.show()
    plt.savefig(options.output,bbox_inches='tight', dpi=100)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
