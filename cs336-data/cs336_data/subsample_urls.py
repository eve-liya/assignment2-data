import gzip
import random

def subsample_urls(input_gz: str, output_file: str, sample_fraction: float = 0.01):
    """
    Reads a gzipped file containing URLs (one URL per line) and writes out a random subset
    of them to an output file.
    
    Args:
        input_gz (str): Path to the input gzipped file.
        output_file (str): Path to the output file.
        sample_fraction (float): Fraction of URLs to retain.
    """
    with gzip.open(input_gz, "rt", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            url = line.strip()
            if random.random() < sample_fraction:
                fout.write(url + "\n")

# This will retain approximately 1% of the URLs.
if __name__ == "__main__":
    subsample_urls("data/enwiki-20240420-extracted_urls.txt.gz", "subsampled_urls.txt", sample_fraction=0.01)
