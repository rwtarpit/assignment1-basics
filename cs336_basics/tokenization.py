"""vocabulary building class for tokenizer"""

import regex as re
import heapq
import multiprocessing
import os
import mmap
from typing import BinaryIO
from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    mini_chunk_size : int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    #mini_chunk_size = 1024*1024*50   Read ahead by 50mb at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))
        
def process_chunk(args : tuple):
    """processes a chunk of text and returns 
    a local pretokenization table."""
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    file_path, start, end, special_tokens = args
    local_table = Counter()
    
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        
            chunk_bytes = mm[start:end]
            
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            
            del chunk_bytes
            
            if special_tokens:
                split_pattern = "|".join(re.escape(token) for token in special_tokens)
                segments = re.split(split_pattern, chunk_text)
            else:
                segments = [chunk_text]
            
            for segment in segments:
                if not segment:  
                    continue

                for match in PAT.finditer(segment):
                    local_table[tuple(bytes(match.group(), "utf-8"))] += 1
                
    return local_table

class GreaterPair:
    def __init__(self, pair_tuple, vocab):
        self.pair = pair_tuple
        self.bytes_a = vocab[pair_tuple[0]]
        self.bytes_b = vocab[pair_tuple[1]]
    
    def __lt__(self, other):
        """change `<` definition for getting 
        lexicographically greater pair from min-heap"""
        return (self.bytes_a, self.bytes_b) > (other.bytes_a, other.bytes_b)

    
class train_tokenizer:
    def __init__(self, vocab_size : int, special_tokens : list):
        self.vocab = {i: bytes([i]) for i in range(256)} #final vocabulary(id:bytes)
        self.vocab_size = vocab_size 
        self.pretokenized_table = {} #unique words and their count in dataset
        self.pair_count_table = {} # count of each byte pair
        self.heap = [] # heap for finding bp for each merge
        self.occurrence_index = {} #set[word ids] of each byte pair
        self.word_id_to_tuple = {} #int id for each unique word {id:word}
        self.word_id_to_count = {} #count of each word id {id:frequency}
        self.merges=[] #merges performed in order
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_tokens = special_tokens
        
        for special_token in self.special_tokens:
            id = max(self.vocab.keys()) + 1
            self.vocab[id] = bytes(special_token,"utf-8")
    
    def pretokenization(self, path : str, num_process : int, seperator : str):
        """pretokenizes the dataset by counting frequency of each unique word.
        
        uses gpt-2's regex to find words and saves them as bytes.
        Splits on special tokens to prevent merging across document boundaries.
        """
        
        with open(path,"rb") as f:
            """opens text file in binary and assigns each process
            with a chunk of text"""
            boundaries = find_chunk_boundaries(f, num_process*3, 1024*1024*40,seperator)
            tasks = []
            for i in range(len(boundaries)-1):
                tasks.append((path, boundaries[i], boundaries[i+1], self.special_tokens))
            
            print(f"Starting pre-tokenization with {len(tasks)} processes...")
            final_counts = Counter() 
            with multiprocessing.Pool(processes=num_process) as pool:
                    results = pool.imap_unordered(process_chunk, tasks)
            
            #final_counts = Counter()
                    for local_table in results:
                        final_counts.update(local_table)
            
            self.pretokenized_table = dict(final_counts)    
            return self
    

    def find_count_table(self):
        """counts frequency for each unique bytepair
        
        allocates unique int id to each word, and then caches
        occurences of each bytepair in the word they appeared.
        """
        current_id = 0
        for word_tuple, frequency in sorted(self.pretokenized_table.items()):
            self.word_id_to_tuple[current_id] = list(word_tuple)
            self.word_id_to_count[current_id] = frequency
            
            for i in range(len(word_tuple)-1):
                byte_pair = (word_tuple[i],word_tuple[i+1])
                self.pair_count_table[byte_pair] = self.pair_count_table.get(byte_pair,0)+frequency
                self.occurrence_index.setdefault(byte_pair, set()).add(current_id)
            current_id+=1   
        
        self.heap = []
        for pair, count in self.pair_count_table.items():
            #bytes_a = self.vocab[pair[0]]
            #bytes_b = self.vocab[pair[1]]
            #negated_tuple = (tuple(255-b for b in bytes_a), tuple(255-b for b in bytes_b))
            heap_entry = (-count, GreaterPair(pair, self.vocab), pair)
            self.heap.append(heap_entry)
        heapq.heapify(self.heap)
        
        return self   
    
    @staticmethod
    def find_bps_from_word(word : list):
        """helper function to find bytepairs in a word with their counts"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i+1]))
        return pairs
    
    def train_one_merge(self):
        """merging logic for one iteration with Lazy Heap Updates."""
        while True:
            if not self.heap:
                return self
            
            neg_count, _, pair = heapq.heappop(self.heap)
            count = -neg_count
            
            # STALE ENTRY CHECK: 
            # If the current count in our master table doesn't match the 
            # count we just popped from the heap, it's an old version. Ignore it.
            actual_count = self.pair_count_table.get(pair, 0)
            if actual_count == count and actual_count > 0:
                max_pair = (pair, count)
                break
        
        # ... [Vocab assignment and merge logging remains the same] ...
        id_a, id_b = max_pair[0]
        new_id = max(self.vocab.keys()) + 1
        self.vocab[new_id] = self.vocab[id_a] + self.vocab[id_b]
        self.merges.append(max_pair[0])
        
        appeared_in_ids = list(self.occurrence_index[max_pair[0]])
        affected_pairs = set()
        
        # --- START SNAPSHOT LOGIC ---
        for word_id in appeared_in_ids:
            word = self.word_id_to_tuple[word_id] 
            freq = self.word_id_to_count[word_id]
            
            # 1. Capture snapshots of pairs before and after
            before_pairs = Counter(zip(word, word[1:]))
            
            # 2. Perform merge
            new_word = []
            i = 0
            while i < len(word): 
                if i < len(word)-1 and word[i] == id_a and word[i+1] == id_b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            self.word_id_to_tuple[word_id] = new_word
            
            after_pairs = Counter(zip(new_word, new_word[1:]))
            
            # 3. Calculate deltas for all affected pairs
            for bp in (set(before_pairs.keys()) | set(after_pairs.keys())):
                diff = after_pairs[bp] - before_pairs[bp]
                if diff != 0:
                    self.pair_count_table[bp] = self.pair_count_table.get(bp, 0) + (diff * freq)
                    # Track this pair to update the heap later
                    affected_pairs.add(bp)
                    
                    if self.pair_count_table[bp] <= 0:
                        self.pair_count_table.pop(bp, None)
                    
                    if after_pairs[bp] > 0:
                        self.occurrence_index.setdefault(bp, set()).add(word_id)
                    else:
                        if bp in self.occurrence_index:
                            self.occurrence_index[bp].discard(word_id)
        # --- END SNAPSHOT LOGIC ---

        # CLEANUP: Remove the pair we just merged
        if max_pair[0] in self.pair_count_table:
            del self.pair_count_table[max_pair[0]]
        if max_pair[0] in self.occurrence_index:
            del self.occurrence_index[max_pair[0]]

        # 5. HEAP UPDATE: Only push the pairs that actually changed
        for bp in affected_pairs:
            cnt = self.pair_count_table.get(bp, 0)
            if cnt > 0:
                # Push the new count. The heap will now have two entries 
                # for 'bp', but our 'while True' loop above will skip the old one.
                heapq.heappush(self.heap, (-cnt, GreaterPair(bp, self.vocab), bp))
             
        #if heap grows too big in size with stale entries       
        if len(self.heap) > 10*len(self.pair_count_table)+100000:
            self.heap=[]
            for pair, count in self.pair_count_table.items():
                heapq.heappush(self.heap,(-count,GreaterPair(pair,self.vocab),pair))
                
        return self
        
    
    def train(self):
        """final train function to generate vocab"""            
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            if not self.pair_count_table:
                print("completed all possible merges before hitting vocab limit!")
                break
            self.train_one_merge()
            #if i%100 == 0:
                #print(f"merge no. {i+1}, vocab size = {len(self.vocab)}")
        return self
    
def train_bpe(input_path : str, vocab_size : int, special_tokens : list[str]):
    
    tokenizer = train_tokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.pretokenization(
        path = input_path,
        num_process = 10,
        seperator=bytes(special_tokens[0],"utf-8")
        )
    #print("pretokenization\n",tokenizer.pretokenized_table)
    tokenizer.find_count_table()
    #print("initial bpe count table\n",tokenizer.pair_count_table)
    #print("initial heap\n", tokenizer.heap)
    tokenizer.train()
    
    vocab = tokenizer.__getattribute__("vocab")
    final_merges = []
    for id_a, id_b in tokenizer.merges:
        byte_a = tokenizer.vocab[id_a]
        byte_b = tokenizer.vocab[id_b]
        final_merges.append((byte_a, byte_b))
    return vocab, final_merges