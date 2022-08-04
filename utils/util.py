import torch
from functools import partial

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    # return [*map_results]
    return tuple(map(list, zip(*map_results)))

