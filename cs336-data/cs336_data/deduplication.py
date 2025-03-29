import os
import hashlib
import string
import random
import hashlib
import unicodedata
from collections import defaultdict
from itertools import combinations

def exact_deduplication(input_paths, output_dir):
    """
    Performs exact line deduplication across multiple input files.
    
    Args:
        input_paths (list): List of file paths to process.
        output_dir (str): Directory to save deduplicated files.
    
    Output:
        Writes deduplicated versions of input files into the output directory.
    """
    # Dictionary to store hash counts
    line_counts = {}

    # First Pass: Count occurrences of each line using a hash
    for file_path in input_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:  # Ignore empty lines
                    line_hash = hashlib.md5(line.encode()).hexdigest()  # Fixed-size hash
                    line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Second Pass: Rewrite files, keeping only unique lines
    for file_path in input_paths:
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        
        with open(file_path, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
            for line in file:
                line = line.strip()
                if line:
                    line_hash = hashlib.md5(line.encode()).hexdigest()
                    if line_counts[line_hash] == 1:  # Only write unique lines
                        out_file.write(line + "\n")

# Union-Find data structure for clustering duplicates.
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    
    def union(self, i, j):
        pi = self.find(i)
        pj = self.find(j)
        if pi != pj:
            self.parent[pj] = pi

def normalize_text(text):
    """
    Normalize text by lowercasing, Unicode NFD normalization,
    stripping accents, removing punctuation, and normalizing whitespace.
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def get_ngrams(text, n):
    """
    Given normalized text, returns a set of word n-grams.
    """
    words = text.split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

def compute_minhash_signature(ngrams, num_hashes):
    """
    Compute a minhash signature for a set of n-grams.
    For each hash function (simulated by seeding MD5 with i), 
    compute the minimum hash value over all n-grams.
    """
    signature = []
    for i in range(num_hashes):
        min_hash = None
        for gram in ngrams:
            # Simulate a distinct hash function using the seed i.
            h = hashlib.md5((str(i) + gram).encode('utf-8')).hexdigest()
            h_int = int(h, 16)
            if min_hash is None or h_int < min_hash:
                min_hash = h_int
        # If document has no ngrams, set signature element to a large value.
        if min_hash is None:
            min_hash = 2**128 - 1
        signature.append(min_hash)
    return signature

def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    """
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def run_minhash_deduplication(input_paths, num_hashes, num_bands, ngram_length, output_dir, jaccard_threshold=0.8):
    """
    Performs fuzzy document deduplication using MinHash and LSH.
    
    Args:
        input_paths (list): List of file paths (each file is one document).
        num_hashes (int): Number of hash functions to compute the MinHash signature.
        num_bands (int): Number of bands to use in LSH (must evenly divide num_hashes).
        ngram_length (int): n-gram length (in words) to use.
        output_dir (str): Directory to write deduplicated documents.
        jaccard_threshold (float): Candidate pair similarity threshold.
    
    Writes:
        For each retained document, writes its original contents (unchanged)
        to the output directory with the same file name.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be evenly divisible by num_bands.")
    rows_per_band = num_hashes // num_bands
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and normalize documents.
    docs = []
    norm_docs = []
    ngram_sets = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
            norm_text = normalize_text(text)
            norm_docs.append(norm_text)
            ngram_sets.append(get_ngrams(norm_text, ngram_length))
    
    num_docs = len(docs)
    
    # Compute minhash signatures for each document.
    signatures = []
    for ng_set in ngram_sets:
        sig = compute_minhash_signature(ng_set, num_hashes)
        signatures.append(sig)
    
    # LSH: For each band, bucket documents by the band signature.
    buckets = defaultdict(list)
    for doc_id, sig in enumerate(signatures):
        for b in range(num_bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(sig[start:end])
            buckets[(b, band_tuple)].append(doc_id)
    
    # Collect candidate duplicate pairs.
    candidate_pairs = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) > 1:
            for i, j in combinations(bucket_docs, 2):
                candidate_pairs.add(tuple(sorted((i, j))))
    
    # Use union-find to cluster duplicates.
    uf = UnionFind(num_docs)
    for i, j in candidate_pairs:
        # Compute true Jaccard similarity between the n-gram sets.
        sim = jaccard_similarity(ngram_sets[i], ngram_sets[j])
        if sim >= jaccard_threshold:
            uf.union(i, j)
    
    # Determine which document to retain from each cluster.
    clusters = defaultdict(list)
    for doc_id in range(num_docs):
        parent = uf.find(doc_id)
        clusters[parent].append(doc_id)
    
    # Randomly select one representative from each cluster.
    kept_docs = set()
    for cluster in clusters.values():
        chosen = random.choice(cluster)
        kept_docs.add(chosen)
    
    # Write out retained documents to the output directory.
    # For each input path, if its corresponding document is retained, write it.
    for i, path in enumerate(input_paths):
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        if i in kept_docs:
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(docs[i])
        else:
            pass
