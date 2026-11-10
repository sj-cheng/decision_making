"""Share immutable geometry, never a team's position-dependent routing caches."""
from functools import lru_cache
import numpy as np
from .core import PublicMap,route_heads


class IsolatedMap(PublicMap):
    def __init__(self,public):
        owner=getattr(public,'_geometry_owner',public)
        # This construction depends only on the announced obstacle map.
        owner._ensure_route_graph()
        self._geometry_owner=owner
        for name in ('cfg','boxes','width','points','free','graph','components','free_indices',
                     '_route_vertices','_route_dist','_vertex_ids'):
            setattr(self,name,getattr(owner,name))
        self._distance=lru_cache(maxsize=128)(self._compute_distance)
        self._route_heads=lru_cache(maxsize=8192)(
            lambda key:route_heads(np.asarray(key),self._route_vertices,self.boxes))
        self._route_tail=lru_cache(maxsize=2048)(
            lambda key:self._route_dist[self._vertex_ids[key]] if key in self._vertex_ids
            else np.min(self._route_dist+self._route_heads(key)[None,:],axis=1))


def isolated_routes(public):
    return IsolatedMap(public)
