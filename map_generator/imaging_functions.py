from pathlib import Path
from typing import Optional

import map_generator.backend_switch as np
from map_generator.backend_switch import scoped
import cv2
import matplotlib.pyplot as plt

def show_map(gen_object, x,y, scale=1, offset=[0,0], sampling = 1, channel=-1, octaves=1,neg_octaves=0, fade=0.5,dim3=False):
    sampling = int(sampling) #Just a check to clean up this input
    #Sampling is used when taking multiple samples per pixel, important with finer noise
    
    
    ### Remap according to scale and offset.
    #Works, but is in need of a cleanup
    #Scale should result in distance per pixel, offset in the new centre
    #x,y are the final image size. Should these affect size of mapped area?
    offset[0] = offset[0]  -x*0.5* scale +0.5* scale
    
    offset[1] = offset[1] -y*0.5* scale +0.5* scale
    offset[0] = offset[0] 
    
    offset[1] = offset[1] 
    ###
    
    output = np.zeros(shape=(x,y))
    if dim3:
        output = np.zeros(shape=(x,y,3))
    
    if channel==-1 and octaves==1 and neg_octaves==0 and fade==0.5:
        extra_args = False
    else:
        extra_args = True
    
    for i in range(x):
        for j in range(y):
            out = 0
            if sampling > 1:
                for _ in range(sampling): #Multiple (jittered) checks per pixel, reduces accuracy at low values on smooth shapes
                    if extra_args:
                        out += gen_object.get_height(i*scale+offset[0]+ (random.random()-0.5)*scale, j*scale + offset[1] + (random.random()-0.5)*scale, channel=channel, octaves=octaves,neg_octaves=neg_octaves, fade=fade)/sampling
                    else:
                        out += gen_object.get_height(i*scale+offset[0]+ (random.random()-0.5)*scale, j*scale + offset[1] + (random.random()-0.5)*scale)/sampling
            else:
                if extra_args:
                    out += gen_object.get_height(i*scale+offset[0], j*scale + offset[1],channel=channel, octaves=octaves,neg_octaves=neg_octaves, fade=fade)
                else:
                    out += gen_object.get_height(i*scale+offset[0], j*scale + offset[1])
            if dim3:
                output[j][i][0] = i
                output[j][i][1] = j
                output[j][i][2] = out
            else:
                output[j][i] = out
    return(output)
def show_map_3d(gen_obj,img_x,img_y,sca,centre_offset=[0,0],num_samples = 1,vectorised = False, **kwargs):
    #x = np.arange(int(centre_offset[0]-0.5*img_x*sca), int(centre_offset[0]-0.5*img_x*sca)+img_x*sca, sca)
    #y = np.arange(int(centre_offset[1]-0.5*img_y*sca), int(centre_offset[1]-0.5*img_y*sca)+img_y*sca, sca)
    x = np.linspace(int(centre_offset[0]-0.5*img_x*sca), int(centre_offset[0]+0.5*img_x*sca), img_x)
    y = np.linspace(int(centre_offset[1]-0.5*img_y*sca), int(centre_offset[1]+0.5*img_y*sca), img_y)
    X, Y = np.meshgrid(x, y)
    if num_samples == 1:
        if not vectorised:
            Z = np.asarray([[gen_obj.get_height(a,b,**kwargs) for a in x] for b in y])
        
        ###VECTORISED SAMPLE OF generator
        else:
            Z = gen_obj.get_height(X,Y,**kwargs)
        
    else:
        Z = np.zeros((img_x,img_y))
        for _ in range(num_samples):
            offset_x = np.random.rand(img_x,img_y)-0.5
            offset_x = offset_x*sca
            offset_x += X
            offset_y = np.random.rand(img_x,img_y)-0.5 
            offset_y = offset_y*sca
            offset_y += Y
            Z += np.asarray([[gen_obj.get_height(a,b,**kwargs) for a in x] for b in y])/num_samples
            
    return (X, Y, Z)

def normalize(Z, output_folder: Optional[Path | str] = None): # normalizes Z to [0,1] and converts to uint16
    minimum_val = np.min(Z)
    maximum_val = np.max(Z)
    Z = Z + -1 * np.min(Z)
    Z = 65535*Z /np.max(Z)
    Z = Z.astype(np.uint16)

    from map_generator.backend_switch import gpu_enabled, jax_available
    if gpu_enabled or jax_available:
        import numpy as np_backup
        Z = np_backup.asarray(to_numpy(Z))  # Convert to numpy array if using GPU backend
    else:
        plt.imshow(Z*0.5 + 0.5)
        plt.show()
    if output_folder is not None:
        output_folder = Path(output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_folder / f"max_{maximum_val}_min_{minimum_val}.png"), Z)
    return Z

def check_layer(sample_slice, layer1=-1, layer2=-1):
    print(sample_slice.shape) 
    print(sample_slice.min(), sample_slice.max())
    ax = plt.imshow(sample_slice[...,layer1,layer2])
    plt.show()

def to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else arr
@scoped
def flatten_negative(z: np.ndarray, threshold=0, weight=1.0) -> np.ndarray:
    """Reduces the impact of negative values in the array.
    At a weight of 1.0, all values below the threshold are set to 0."""
    fraction_normal = 1-0.5*weight
    fraction_absolute = 0.5*weight
    z = z - threshold
    z = fraction_normal*z+fraction_absolute*np.abs(z)
    z = z + threshold
    return z