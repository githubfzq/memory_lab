from neurom.core.morphology import Morphology
from functools import wraps
import shelve
from hashlib import sha1
import pickle

class Cache:
    def __init__(self, db='cache.db') -> None:
        self.db = db

    def shelve_cache(self, func):
        @wraps(func)
        def wrapper_shelve_cache(*args, **kwargs):
            hash_val = self.get_hash_key(func.__name__)
            hash_val = self.get_hash_key(args, hash_val)
            hash_val = self.get_hash_key(kwargs, hash_val)
            key = hash_val.hexdigest()
            with shelve.open(self.db) as db:
                if key in db:
                    return db[key]
                else:
                    result = func(*args, **kwargs)
                    db[key] = result
                    return result
        return wrapper_shelve_cache

    @staticmethod
    def get_hash_key(value, sha_value=None):
        from morphological_functions import morpho_parser
        from patch_clamp_functions import electro_parser
        from neuron_cluster_analysis import cluster_processor

        s = sha_value or sha1(b'')
        if isinstance(value, str):
            s.update(value.encode())
            return s
        elif isinstance(value, (int, bool)):
            s.update(bytes(value))
            return s
        elif value is None:
            return s
        elif isinstance(value, (list, tuple)):
            for v in value:
                s = Cache.get_hash_key(v, s)
            return s
        elif isinstance(value, dict):
            for k, v in value.items():
                s = Cache.get_hash_key(k, s)
                s = Cache.get_hash_key(v, s)
            return s
        elif isinstance(value, (morpho_parser, electro_parser)):
            return Cache.get_hash_key(value.path_root, s)
        elif isinstance(value, Morphology):
            return Cache.get_hash_key(value.name, s)
        elif isinstance(value, cluster_processor):
            return Cache.get_hash_key((value.mpp.path_root, value.elec_ps.path_root), s)
        else:
            s.update(pickle.dumps(value))
            return s

    def get_cache_info(self):
        result = {}
        with shelve.open(self.db) as db:
            for k,v in db.items():
                result[k] = v
        return result

    def clear_cache(self):
        with shelve.open(self.db) as db:
            for k in db:
                del db[k]

cache = Cache()