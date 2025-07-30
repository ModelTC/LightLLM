import xxhash
import numpy as np
from typing import List, Tuple, Optional

def compute_token_list_hash(self, tokens: List[int], chuncked_size:int) -> List[int]:
    chunks_hash_value = []
    hsum = xxhash.xxh3_64()

    chuncked_size = (len(tokens) + chuncked_size - 1)  // chuncked_size
    
    for i in range(chuncked_size):
        start_index = i * chuncked_size
        end_index = (i + 1) * chuncked_size
        chunk = tokens[start_index:end_index]
        chunk_np = np.array(chunk, dtype=np.uint64)
        hsum.update(chunk_np.tobytes())
        
        hash_value = hsum.intdigest()
        chunks_hash_value.append(hash_value)
        
    return chunks_hash_value