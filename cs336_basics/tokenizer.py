"""tokenizer class for encoding and decoding of text using prebuilt vocabulary"""

import pickle
import regex as re
import cProfile
import pstats
import multiprocessing
import mmap
from typing import Iterable, Iterator
import array

from tokenization import find_chunk_boundaries

profiler = cProfile.Profile()

worker_tokenizer = None
#worker_pattern = None
worker_cache = {}

def init_worker(vocab, merges, special_tokens):
    global worker_tokenizer, worker_cache
    # This runs ONCE when a worker process starts
    worker_tokenizer = tokenizer(vocab, merges, special_tokens)
    #worker_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    worker_cache = {}
    
def encode_chunk(chunk_data):
    """
    Worker function for each process to encode chunk of text
    """
    file_path, start, end = chunk_data
    
    with open(file_path,"rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        
            chunk_bytes = mm[start:end]
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            del chunk_bytes
    
    return worker_tokenizer.encode(chunk_text, worker_cache)
            
    

class tokenizer():
    def __init__(self, vocab : dict[int,bytes], merges : list[tuple[bytes,bytes]], special_tokens = None):
        self.decoding_vocab : dict = vocab #{id:tuple}
        self.encoding_vocab : dict = {value : key for key,value in vocab.items()}  #{tuple:id}
        self.merges : dict = {merge:i for i, merge in enumerate(merges)}
        self.special_tokens : list = special_tokens
        #self._cache = {}
        self.PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
        if special_tokens is not None:
            for special_token in self.special_tokens:
                if bytes(special_token,"utf-8") not in self.encoding_vocab.keys():
                    id = max(self.decoding_vocab.keys())+1
                    self.decoding_vocab[id] = bytes(special_token,"utf-8")
                    self.encoding_vocab[bytes(special_token,"utf-8")] = id
                
    @classmethod
    def from_files(cls,vocab_filepath : str, merge_filepath : str, special_tokens : list[str]|None = None):
        
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        
        with open(merge_filepath,"rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab,merges,special_tokens)
        

    def encode_token(self, token_bytes : list[bytes]):
        """merges bytes based on order of merges during vocab training"""
        def get_pairs(token_bytes):
            pairs = set()
            el_1 = token_bytes[0]
            for el_2 in token_bytes[1:]:
                pairs.add((el_1,el_2))
                el_1 = el_2
            return pairs
        
        byte_pairs = get_pairs(token_bytes)
        if not byte_pairs:
            return token_bytes
        
        while True:
            first_merge = min(byte_pairs, key = lambda merge : self.merges.get(merge,float('inf')))
            if first_merge not in self.merges:
                break
            
            new_token_bytes = []
            i = 0
            while i < len(token_bytes): 
                if i < len(token_bytes)-1 and token_bytes[i] == first_merge[0] and token_bytes[i+1] == first_merge[1]:
                    new_token_bytes.append(first_merge[0]+first_merge[1])
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1
            
            token_bytes = new_token_bytes
            if len(token_bytes) == 1:
                break
            else:
                byte_pairs = get_pairs(token_bytes)
        
        return token_bytes
        
        
    
                
    def encode(self, text, cache = {}):
        """
        Optimized encoding: Processes text in chunks and yields IDs immediately 
        to prevent memory spikes.
        """
        # Pre-compile patterns outside the loop
        protected_part = "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) if self.special_tokens else None
        
        final_encodings = []

        def process_and_encode_segment(token):
            # Cache for sub-word encodings to avoid re-merging the same word 1000 times
            # Zipf's law
            if len(cache)>200000:
                cache.clear()
            
            if token in cache:
                return cache[token]
                
            if self.special_tokens and token in self.special_tokens:
                res = [self.encoding_vocab[token.encode("utf-8")]]
            else:
                # Byte-level BPE merge logic
                byte_list = [bytes([b]) for b in token.encode("utf-8")]
                byte_list = self.encode_token(byte_list)
                res = [self.encoding_vocab[b] for b in byte_list]
            
            cache[token] = res
            return res

        # Stream segments directly into the list
        if not self.special_tokens:
            for match in self.PATTERN.finditer(text):
                final_encodings.extend(process_and_encode_segment(match.group()))
        else:
            parts = re.split(f"({protected_part})", text)
            for part in parts:
                if not part: continue
                if part in self.special_tokens:
                    final_encodings.extend(process_and_encode_segment(part))
                else:
                    for match in self.PATTERN.finditer(part):
                        final_encodings.extend(process_and_encode_segment(match.group()))
                        
        return final_encodings
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, 
        lazily yields token IDs one by one.
        """
        for text_block in iterable:
            # Encode the current block (e.g., a line from a file)
            tokens = self.encode(text_block)
            
            # Yield each token ID individually to maintain the generator contract
            for token_id in tokens:
                yield token_id
        
                    
    def decode(self, ids: list):
        """decodes bytes back to string.
        
        recursively decodes bytes back in original
        vocabulary range(0-255) and then coverts back
        to text.
        """
        raw_bytes = []
        
        def unpack(id):
            val = self.decoding_vocab[id]
            if isinstance(val, tuple):
                return unpack(val[0]) + unpack(val[1])
            else:
                return val

        for token_id in ids:
            raw_bytes.extend(unpack(token_id))
        print("raw bytes",raw_bytes)
        raw_bytes = [ord(byte)  if isinstance(byte,str) else byte for byte in raw_bytes]
        return bytes(raw_bytes).decode("utf-8",errors="replace")
    



def parellel_encoding(file_path, data_path, num_process, separator):
    
    with open(data_path,"rb") as f:
        """opens vocab and merges file"""
        loaded_data = pickle.load(f)
        
    # Initialize tokenizer 
    init_args = (loaded_data['vocab'], loaded_data['merges'], ["<|endoftext|>"])
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process*20, 1024*1024*10, separator)
        # Note: No 'tok' in tasks anymore
    tasks = [(file_path, boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        
        # Use 'initializer' to setup the tokenizer once per worker
    with open(r"C:\Users\Arpit Rawat\Desktop\cs336\assignment1-basics\cs336_basics\data\owt_train_encoded.bin", "wb") as f:
        with multiprocessing.Pool(processes=num_process, 
                                initializer=init_worker, 
                                initargs=init_args) as pool:
            for result in pool.imap(encode_chunk,tasks):
                f.write(array.array('H',result).tobytes())
            
    #final_encodings = [token for chunk in results for token in chunk]
    #return final_encodings
    

data_path = r"C:\Users\Arpit Rawat\Desktop\cs336\assignment1-basics\cs336_basics\owt_train.pkl"
file_path = r"C:\Users\Arpit Rawat\Desktop\cs336\assignment1-basics\cs336_basics\data\owt_train.txt"

special_tokens = ["<|endoftext|>"]


if __name__ == "__main__":
    profiler.enable()
    encodings = parellel_encoding(file_path,data_path,num_process=12,separator=bytes(special_tokens[0],"utf-8"))

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

          
    