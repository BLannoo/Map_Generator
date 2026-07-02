import map_generator.backend_switch as np
from map_generator.noise_functions import my_perl, blend_distance_layers
import random
import math
import gc
from map_generator.imaging_functions import flatten_negative


class landscape_gen():
    def __init__(self,lat = 10, long = 10, num_plates = 5, boundaries = True):
        self.lat = lat 
        self.long = long 
        self.lin_sca = math.sqrt(lat*long)
        self.noise_negative_octaves = np.log2(self.lin_sca).astype(int)-4 # 4 works for 200km, each doubling above this doubles the range
        self.mountains_done = False 
        self.rivers_done = False 
        self.centroids = []
        self.heights =[]
        slopes_r = []
        slopes_theta = []
        if boundaries:
            self.centroids = [np.asarray([0,lat]), np.asarray([long,0]),
                              np.asarray([long, lat]), np.asarray([0,0]),
                              np.asarray([0,lat*0.5]), np.asarray([long*0.5,0]),
                              np.asarray([long*0.5, lat]), np.asarray([long, lat*0.5])] #Seafloor tiles on each corner and middles of edges. Redundant if world is made round
            self.heights = [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]
            slopes_r = [0,0,0,0,0,0,0,0]
            slopes_theta = [0,0,0,0,0,0,0,0]
        self.mountains = []
        self.mountains_tiles_x = []
        self.mountains_tiles_y = []
        for i in range (num_plates): 
            new_xy = np.asarray([random.random()*long, random.random()*lat])
            self.centroids.append(new_xy)
            max_height = np.divide(new_xy, np.asarray([long, lat]))
            max_height = np.multiply(max_height, 1 - max_height)
            max_height = np.sum(max_height)
            self.heights.append(max(random.random(),random.random())*4*max_height-1) #To do: scale heights so they're higher in the middle and lower near edges
            slopes_r.append(random.random()**2) # Bias towards lower slopes
            slopes_theta.append(random.random()*2*np.pi)
        self.centroids = np.asarray(self.centroids)
        self.slopes_x = np.multiply(np.cos(np.asarray(slopes_theta)), np.asarray(slopes_r))
        self.slopes_y = np.multiply(np.sin(np.asarray(slopes_theta)), np.asarray(slopes_r))
        self.river_density = 20

    def compute_offsets(self, x, y,offs = 1, fine_offs =1,neg_octave=4, **kwargs): #consider using a different function for fine_offset
        
        # neg_octave = int(np.log2(self.lin_sca)-4) # 4 works for 200km, each doubling above this doubles the range
        # neg_octave = self.noise_negative_octaves
        offset = my_perl.sample(x,y,neg_octaves = neg_octave, octaves=-1,ndims=2)  ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        # print(offset.shape)
        fine_offset = my_perl.sample(x,y,neg_octaves = 1, octaves = 4,ndims=3) #Use ndims=3 for hill noise?
        offset =  np.add(fine_offset[:,:,0:2],offset) * offs
        fine_offset = np.add(fine_offset,my_perl.sample(x,y,neg_octaves = -4, octaves = 8,ndims=3)) * fine_offs
        fine_x = np.add(x, fine_offset[:,:,0])
        fine_y = np.add(y, fine_offset[:,:,1])
        #fine_offset = fine_offset[:,:,2] use if a third dimension is needed
        coarse_x = x+offset[:,:,0]
        coarse_y = y+offset[:,:,1]
        return(coarse_x, coarse_y, fine_x, fine_y)
    def get_rivers(self,x,y,weight=1, lacunarity=1.414, **kwargs):
        freq = self.river_density/self.lin_sca #Frequency of major rivers, 20 across the world (nb not necessarily 20 separate rivers, but 20 points at which they're defined)
        river_z = my_perl.dendry(x=x,y=y, intensity=weight, dendry_layers=5, upres=2, final_sample=10, 
                                 initial_method='b', upres_tier_max=0,
                                 base_frequency=freq, epsilon=0.45, skew=0.45, lacunarity=lacunarity, 
                                 push_upstream=0.1, push_downstream=0.1,
                                 scale_factor_start=0.75, soften_start = 0.75, weight_t=0.0, 
                                 bias_value = 2, verbose=False, control_function = self.get_base_height,
                                 scale = freq, blend_scale = 1,return_full=True,
                                 include_secondary=False, **kwargs)
        return(river_z)
    def get_base_height(self, x,y, include_secondary=True,distance_cap = 8,slope_intensity=1, **kwargs):
        ### Updated for higher dimensional inputs
        # tested with 2 and 4D inputs, theoretically should handle any ndims, but best to test broadcasting first for new applications
        # currently contains many assert statements to check this, but these can likely be removed
        x = np.array(x)
        y = np.array(y)
        input_ndims = x.ndim
        assert y.ndim == input_ndims, "X and Y must have the same number of dimensions"
        input_shape = x.shape
        assert y.shape == input_shape, "X and Y must have the same shape"
        n_plates = len(self.centroids)
        shape_with_plates = input_shape + (n_plates,)
        #noise_height = offset[2] #Necessary? May cause tiling depending on noise
        diff_x = np.subtract(x[..., np.newaxis], self.centroids[:, 0]) #Distance from each point to each plate
        diff_y = np.subtract(y[..., np.newaxis], self.centroids[:, 1]) #Distance from each point to each plate
        
        
        # calculate slope height contribution based on this linear distance
        base_height = np.multiply(diff_x, self.slopes_x[np.newaxis, ...])  
        # hardcoding of all additional dimensions is seemingly not possible while maintaining flexibility without for loop or braching eg if statements
        # instead one new axis is added to enforce broadcasting
        base_height += np.multiply(diff_y, self.slopes_y[np.newaxis, ...])

        base_height *= slope_intensity/self.lin_sca #Scale the slope height contribution such that it would vary by a total of slope_intensity across the world
        diff_x = np.power(diff_x, 2) #Square the distance to each plate
        diff_y = np.power(diff_y, 2) #Square the distance to each plate

        distances = np.sqrt(np.add(diff_x, diff_y)) #Calculate the distance to each plate

        #plate_num = np.argmin(distances, axis = -1)
        # #Distance to closest plate
        plate_dist = np.min(distances, axis = -1) # shape agnostic version
        # assert plate_dist.shape == input_shape, "Plate distance shape does not match input shape"
        
        distances = np.subtract(distances, plate_dist[...,np.newaxis]) #How much further a plate is than the closest plate
        assert distances.shape == shape_with_plates, "Distances shape does not match expected shape with plates"
        if include_secondary: #Functionally use twice the distance for this
            distances_2 = np.minimum(distances, 2*self.lin_sca/distance_cap) * 0.5*distance_cap/self.lin_sca
            distances_2 = np.clip(1 - distances_2,0,1)
            distances_2 = 3*distances_2**2 - 2*distances_2**3
        distances = np.minimum(distances, self.lin_sca/distance_cap) * distance_cap/self.lin_sca #Cap the distance at 10%? of the world size, ranging from 0 to 1
        
        distances = np.clip(1 - distances,0,1) #Weightings of each plate, this is effectively a blurring as you move away and causes primary ridges
        distances = 3*distances**2 - 2*distances**3 #smoothstep
        assert distances.shape == shape_with_plates, "Distances shape does not match expected shape with plates after processing"
        base_height += np.asarray(self.heights)[np.newaxis, :]
        base_height += np.multiply(distances, base_height) # multiplying in a separate step so that the slope contribution is also scaled
        base_height = np.sum(base_height, axis = -1)
        base_height = base_height - np.sum(np.asarray(self.heights)) #For an unknown reason, there seems to be global influence of all plates, therefore this correction is needed
        # TODO: identify why this global influence occurs and whether it can be mitigated in a more principled way

        
        # This point in the process gives a very old, eroded landscape, similar to canyons or the Blue Mountains
        if include_secondary:
            #Secondary shape should result in a curve that dips negative, making negative plates cause a ridge on neighbours, mimicking subduction
            distances = np.multiply(distances_2, np.cos(distances_2*1.5*math.pi)) #Current secondary curve method, can be replaced.
            distances = np.sum(np.multiply(distances, np.asarray(self.heights)[np.newaxis, :]), axis = -1)
            return(base_height, distances)
        else:
            return base_height
    
    def get_mountain_heights(self, x,y,weight, bias = -0.175, **kwargs):
        #output = 0.125*distances#tbd, importance of the boundary itself, to generally change elevation rather than just mountain creation
        
        neg_octave = 3 # int(np.log2(self.lin_sca)-4) # 3 works for 200km, each doubling above this doubles the range
        output = np.multiply(np.maximum(weight, 0.0) , my_perl.sample(x,y,voron=True,neg_octaves=neg_octave,octaves=2,ndims=1,fade=0.3)[:,:,0])
        output = 0.4* output*(0.5**neg_octave) + bias
        output =  np.maximum(output, 0)#Add mountain texture
        #base_height += np.multiply(2 ** (-1 * base_height**2), fine_offset)*0.125 #Add hills
        return (output)
    
    def sample_nearby(self,x,y,scale = 1, sample_depth=5, **kwargs): # Not yet in use, may be useful eg for blurring or antialiasing
        output = np.zeros((x.shape[0],x.shape[1],sample_depth))
        for i in range(sample_depth):
            offx = np.sin(i)*scale*0.001
            offy = np.cos(i)*scale*0.001
            output[:,:,i] = self.get_base_height(x+offx,y+offy,**kwargs)/sample_depth
        return(output)

    def layerise(self,X,Y,Z, freq = 30, weight = 0.25, **kwargs):
        layer_noise = my_perl.sample(X,Y,neg_octaves = 3, octaves = 1,ndims=3)/8
        layer_input = Z * (2-np.absolute(layer_noise[:,:,0]))
        #layer_input += np.multiply(X, layer_noise[:,:,1])
        #layer_input += np.multiply(Y, layer_noise[:,:,2])
        layer_input = layer_input*freq#number of layers to have per kilometer
        layer_output = np.sin(layer_input) + 0.35*np.sin(3*layer_input)+ 0.2*np.sin(5*layer_input)
        Z += weight*layer_output/freq
        return(Z)

    def get_height(self, x,y,mountainsca = 1,riversca=500,rivernoise=0.2,neg_octave = 4, **kwargs):
        # neg_octave = 4# int(np.log2(self.lin_sca)-4) # 4 works for 200km, each doubling above this doubles the range
        coarse_x, coarse_y, fine_x, fine_y = self.compute_offsets(x,y,neg_octave = neg_octave, **kwargs) 
        #are fine_x and fine_y necessary for anything besides mountains?
        if mountainsca == 0 and riversca == 0:
            del fine_x, fine_y
            gc.collect()

        base, secondary = self.get_base_height(coarse_x, coarse_y,**kwargs)
        # del coarse_x, coarse_y
        # gc.collect()
        if mountainsca > 0:
            mountains = self.get_mountain_heights(fine_x, fine_y, np.clip(base*0.5-0.25, 0,1),**kwargs)*200*mountainsca/(self.lin_sca)
            # del fine_x, fine_y
            # gc.collect()
            # layered = self.layerise(x,y,base+mountains, weight=0)
        else:
            mountains = np.zeros_like(x)
            # layered = self.layerise(x,y,base, weight=0)
        if riversca > 0:
            freq = self.river_density/self.lin_sca
            lacunarity = 1.618
            rivers_full = self.get_rivers(coarse_x*rivernoise+x*(1-rivernoise),coarse_y*rivernoise+y*(1-rivernoise),weight = np.clip((base+secondary+mountains)*0.85, 0, 1), lacunarity=lacunarity, ** kwargs)*freq*0.1
            river_z = blend_distance_layers(rivers_full, intensity = np.clip((base+secondary)*0.85, 0, 1), 
                                            lacunarity=lacunarity, bias_value=0.0025,base_frequency=6*np.sqrt(freq))*freq*riversca*2
            river_z = np.power(river_z, 0.5)
            # river_map = 1 - blend_distance_layers(rivers_full, intensity = np.clip(np.abs(layered*0.5), 0, 1), 
            #                                 lacunarity=lacunarity, bias_value=0.0,base_frequency=3*np.sqrt(freq))
        else:
            rivers_full = np.zeros((x.shape[0], x.shape[1], 1))
            river_z = np.zeros_like(base)
            river_map = np.zeros_like(base)
        
        base_0 = flatten_negative(base, threshold=0.0, weight=0.9)
        river_0 = np.multiply(river_z, base_0)
        return (base, mountains , base_0+river_0+mountains+secondary*0.4, river_z, secondary)