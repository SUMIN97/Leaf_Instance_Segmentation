import numpy as np
import torch
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from PIL import Image
import math
import torch.nn.functional as F

def get_seeds(dist_map, seeds_thres=0.7, seeds_min_dis=5):
    dist_map = np.squeeze(dist_map)  # (h,w)
    coordinates  = peak_local_max(dist_map, min_distance=seeds_min_dis, threshold_abs=seeds_thres * dist_map.max())
    mask = np.zeros_like(dist_map)
    for coord in coordinates:
        mask[coord[0], coord[1]] = 1.0
    return mask

def remove_noise(l_map, d_map, min_size=10, min_intensity=0.1):
    max_instensity = d_map.max()
    props = regionprops(l_map, intensity_image=d_map)
    for p in props:
        if p.area < min_size:
            l_map[l_map==p.label] = 0
        if p.mean_intensity/max_instensity < min_intensity:
            l_map[l_map==p.label] = 0
    return label(l_map)

def mask_from_seeds(efeat, seeds, similarity_thres):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seeds = label(seeds).astype('uint8')
    props = regionprops(seeds)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        emb_mean = torch.mean(efeat[row, col], dim=0)
        emb_mean = F.normalize(emb_mean, p=2, dim=0)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        #numpy
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]

        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()

        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds

def mask_from_seeds_v2(efeat, seeds, similarity_thres, distmap):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seeds = label(seeds).astype('uint8')
    props = regionprops(seeds)

    inverse_distmap = np.ones_like(distmap)* distmap.max() - distmap
    inverse_distmap = np.expand_dims(inverse_distmap,2)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        key_feat = efeat[row, col] * inverse_distmap[row,col]
        key_feat = torch.mean(key_feat, dim=0)
        emb_mean = F.normalize(key_feat, p=2, dim=0)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        #numpy
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]

        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()


        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds