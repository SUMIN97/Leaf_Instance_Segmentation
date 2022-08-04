from pathlib import Path
import pickle
import random

data_png_path = Path('/home/lab/ssd2/Plant/Code/Leaf_Instance_Segmentation/data/cvppp/train')
neighbor_path = Path('/home/lab/ssd2/Plant/Code/PEIS/data/neighbors_dilation10')
distance_map_path = Path('/home/lab/ssd2/Plant/Code/PEIS/data/dismap')
data_dict = {}
val = {}
#key pattern example : A1_plant001

#rgb, label, center
for folders_path in data_png_path.iterdir():
    a_folder = folders_path.stem
    plants = []
    for path in folders_path.iterdir():
        if path.suffix == '.csv' : continue

        plant, role = path.stem.split('_')
        if role == 'fg' or role == 'centers': continue

        if plant in plants:
            data_dict[a_folder+'_'+plant].update({role : str(path)})
        else:
            data_dict[a_folder+'_'+plant] = {role: str(path)}
            plants.append(plant)

#neighbor
for path in neighbor_path.iterdir():
    plant = path.stem
    data_dict[plant].update({'neighbor':str(path)})

#distance_map
for path in distance_map_path.glob("*/*.png"):
    a_folder = path.parts[-2]
    plant = path.stem
    data_dict[a_folder+'_'+plant].update({'distance_map':str(path)})

#Split train and val
val_plants = random.sample(list(data_dict.keys()), 10)
for p in val_plants:
    d = data_dict.pop(p)
    val[p] = d

with open('/home/lab/ssd2/Plant/Code/Leaf_Instance_Segmentation/data/train.pickle', 'wb') as f:
    pickle.dump(data_dict, f)

with open('/home/lab/ssd2/Plant/Code/Leaf_Instance_Segmentation/data/val.pickle', 'wb') as f:
    pickle.dump(data_dict, f)




