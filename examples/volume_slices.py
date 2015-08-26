#! /usr/bin/env python


import numpy as np
import numpy.ma as ma

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
    
    parser.add_argument('input',    help="Input image")
    parser.add_argument('--overlay',help="Input overlay image")
    parser.add_argument('output',   help="Output image")
    
    parser.add_argument('--range', help="Image range",nargs=2,type=float)
    parser.add_argument('--orange',help="Overlay range",nargs=2,type=float)
    parser.add_argument('--cmap',  help="Image color-map",default='gray')
    parser.add_argument('--ocmap', help="Overlay color-map",default='jet')
    parser.add_argument('--slices',help="Number of slices to extract", default=10, type=int)
    parser.add_argument('--title', help="Title of image")
    parser.add_argument('--dpi',   help="Figure DPI",default=100,type=float)
    parser.add_argument('--obg',   help="Set overlay background",type=float)
    
    parser.add_argument('--ialpha', help="Image alpha",default=0.8,type=float)
    parser.add_argument('--oalpha', help="Overlay alpha",default=0.2,type=float)
    parser.add_argument('--max',    help="Use max mixing instead of alpha blending",default=False,action="store_true")
    options = parser.parse_args()
    
    return options

def alpha_blend(si, so, ialpha, oalpha):
    """Perform alpha-blending
    """
    si_rgb =   si[..., :3]
    si_alpha = si[..., 3]*ialpha
    
    so_rgb =   so[..., :3]
    so_alpha = so[..., 3]*oalpha
    
    out_alpha = si_alpha + so_alpha * (1. - si_alpha)
    
    out_rgb = (si_rgb * si_alpha[..., None] +
        so_rgb * so_alpha[..., None] * (1. - si_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros_like(si)
    out[..., :3] = out_rgb 
    out[..., 3]  = out_alpha
    
    return out


def max_blend(si,so):
    """Perform max-blending
    """
    return np.maximum(si,so)


def register_custom_maps():
    # register custom maps
    plt.register_cmap(cmap=colors.LinearSegmentedColormap('red',
        {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         
         'alpha':  ((0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0))         
        }))
         
    plt.register_cmap(cmap=colors.LinearSegmentedColormap('green', 
        {'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'alpha': ((0.0, 0.0, 1.0),
                   (1.0, 1.0, 1.0))         
        }))

    plt.register_cmap(cmap=colors.LinearSegmentedColormap('blue', 
        {'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         
         'alpha': ((0.0, 0.0, 1.0),
                   (1.0, 1.0, 1.0))         
        }))


if __name__ == "__main__":
    options = parse_options()
    register_custom_maps()
    
    _img=minc.Image(options.input)
    _idata=_img.data
    data_shape=_img.data.shape
    
    _ovl=None
    _odata=None
    omin=0
    omax=1
    
    if options.overlay is not None:
        _ovl=minc.Image(options.overlay)
        if _ovl.data.shape != data_shape:
            print("Overlay shape does not match image!\nOvl={} Image={}",repr(_ovl.data.shape),repr(data_shape))
            exit(1)
        if options.orange is None:
            omin=np.nanmin(_ovl.data)
            omax=np.nanmax(_ovl.data)
        else:
            omin=options.orange[0]
            omax=options.orange[1]
        _odata=_ovl.data
        
        if options.obg is not None:
            _odata=ma.masked_less(_odata, options.obg)
        
    slices=[]
    
    # setup ranges
    vmin=vmax=0.0
    if options.range is not None:
        vmin=options.range[0]
        vmax=options.range[1]
    else:
        vmin=np.nanmin(_idata)
        vmax=np.nanmax(_idata)

    cm = plt.get_cmap(options.cmap)
    cmo= plt.get_cmap(options.ocmap)
    cmo.set_bad('k',alpha=0.0)

    samples=options.slices

    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    oNorm  = colors.Normalize(vmin=omin, vmax=omax)
    
    scalarMap  = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    oscalarMap = cmx.ScalarMappable(norm=oNorm, cmap=cmo)

    for i in range(0,data_shape[0],data_shape[0]/(samples-1)):
        si=scalarMap.to_rgba(_idata[i , : ,:])

        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[i , : ,:])
            if options.max: si=max_blend(si,so)
            else: si=alpha_blend(si,so, options.ialpha, options.oalpha)
            
        slices.append( si )
        
    for i in range(0,data_shape[1],data_shape[1]/(samples-1)):
        si=scalarMap.to_rgba(_idata[: , i ,:])

        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[: , i ,:])
            if options.max: si=max_blend(si,so)
            else: si=alpha_blend(si,so, options.ialpha, options.oalpha)

        slices.append( si )
    
    for i in range(0,data_shape[2],data_shape[2]/(samples-1)):
        si=scalarMap.to_rgba(_idata[: , : , i])

        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[: , : , i])
            if options.max: si=max_blend(si,so)
            else: si=alpha_blend(si,so, options.ialpha, options.oalpha)

        slices.append( si )
    
    w, h = plt.figaspect(3.0/samples)
    fig = plt.figure(figsize=(w,h))
    
    #outer_grid = gridspec.GridSpec((len(slices)+1)/2, 2, wspace=0.0, hspace=0.0)
    
    for i,j in enumerate(slices):
        ax =  plt.subplot2grid( (3, samples), (i/samples, i%samples) )
        imgplot = ax.imshow(j,origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_visible(False)
        
    if options.title is not None:
        plt.suptitle(options.title,fontsize=20)
        plt.subplots_adjust(wspace = 0.0 ,hspace=0.0)
    else:
        plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)
    
    #fig.tight_layout()
    #plt.show()
    plt.savefig(options.output, bbox_inches='tight', dpi=options.dpi)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
