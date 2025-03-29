from pathlib import Path
from warcio.archiveiterator import ArchiveIterator
import gzip
import random
from cs336_data import extract_text, quality_classifier

def create_quality_dataset(
    positive_warc: str | Path,
    negative_warc: str | Path,
    output_file: str | Path,
    min_word_count: int = 50,
    language: str = "en"
) -> None:
    """
    Create a balanced fastText training dataset from two WARC files with enhanced filtering.
    
    Positive examples (high quality) are labeled as __label__high.
    Negative examples (low quality) are labeled as __label__low.
    
    Args:
        positive_warc (str or Path): Path to the high-quality WARC file.
        negative_warc (str or Path): Path to the low-quality WARC file.
        output_file (str or Path): Path to write the combined training dataset.
        min_word_count (int): Minimum number of words a document must have to be included.
        language (str): ISO language code to filter for (default: "en").
    """
    import identify_text
    
    # Store processed examples for balancing later
    positive_examples = []
    negative_examples = []
    
    def process_warc(warc_path: str | Path, label: str, examples_list, apply_quality_filters=True) -> None:
        count = 0
        with gzip.open(str(warc_path), "rb") as stream:
            for record in ArchiveIterator(stream):
                # Extract text from the record
                html_bytes = record.content_stream().read()
                text = extract_text(html_bytes)
                if not text:
                    continue
                    
                # Clean the text (remove extra whitespace/newlines)
                clean_text = " ".join(text.split())
                
                # Only include documents with a sufficient number of words
                if len(clean_text.split()) < min_word_count:
                    continue
                if apply_quality_filters:
                    # Apply Gopher quality filters ONLY for positive examples
                    if not quality_classifier.gopher_quality_filters(clean_text):
                        continue
                    
                    # Check language (if specified) - apply to both positive and negative
                    if language:
                        detected_lang, confidence = identify_text.identify_language(clean_text[:1000])
                        if detected_lang != language or confidence < 0.5:
                            continue
                    
                    # Filter out NSFW content - apply to both positive and negative
                    nsfw_label, nsfw_conf = identify_text.identify_nsfw(clean_text[:1000])
                    if nsfw_label == "nsfw" and nsfw_conf > 0.7:
                        continue
                    
                    # Filter out toxic content - apply to both positive and negative
                    toxic_label, toxic_conf = identify_text.identify_hatespeech(clean_text[:1000])
                    if toxic_label == "toxic" and toxic_conf > 0.7:
                        continue
                    
                # Store valid examples
                examples_list.append(clean_text)
                count += 1
                
                # Print progress periodically
                if count % 100 == 0:
                    print(f"Processed {count} valid {label} examples")
                if count == 1200:
                    break
        
    
    print("Processing positive (high quality) examples...")
    process_warc(positive_warc, "high", positive_examples, apply_quality_filters=True)
    
    print("Processing negative (low quality) examples...")
    process_warc(negative_warc, "low", negative_examples, apply_quality_filters=False)
    
    # Balance the datasets by sampling
    print(f"Found {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    target_size = min(len(positive_examples), len(negative_examples))
    
    if len(positive_examples) > target_size:
        positive_examples = random.sample(positive_examples, target_size)
    if len(negative_examples) > target_size:
        negative_examples = random.sample(negative_examples, target_size)
    
    print(f"Balanced to {target_size} examples per class")
    
    # Write the balanced dataset
    with open(output_file, "w", encoding="utf-8") as out_f:
        for example in positive_examples:
            out_f.write(f"__label__high {example}\n")
        for example in negative_examples:
            out_f.write(f"__label__low {example}\n")
    
    print(f"Quality dataset created at: {output_file}")
    print(f"Total examples: {len(positive_examples) + len(negative_examples)}")

if __name__ == '__main__':
    # Define file paths.
    positive_warc_file = "data/subsampled_positive_urls.warc.warc.gz"
    negative_warc_file = "data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz"
    training_dataset = "data/quality-dataset-train.txt"
    
    # Step 1: Create the training dataset.
    create_quality_dataset(positive_warc_file, negative_warc_file, training_dataset, min_word_count=50)
