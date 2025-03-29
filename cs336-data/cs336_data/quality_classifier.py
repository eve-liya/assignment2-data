from pathlib import Path
import re
from pathlib import Path
from fasttext import train_supervised, load_model

def gopher_quality_filters(text: str) -> bool:
    """
    Applies Gopher quality filters to a given text.
    
    Rules:
    - Document must contain between 50 and 100,000 words.
    - Mean word length must be between 3 and 10 characters.
    - No more than 30% of lines end with an ellipsis ("...").
    - At least 80% of words must contain at least one alphabetic character.
    
    Returns True if the text passes all filters, False otherwise.
    """
    # Tokenize text into words using whitespace splitting.
    words = re.findall(r'\S+', text)
    num_words = len(words)
    
    # Rule 1: Word count between 50 and 100,000.
    if num_words < 50 or num_words > 100000:
        return False

    # Rule 2: Mean word length between 3 and 10 characters.
    total_chars = sum(len(word) for word in words)
    mean_word_length = total_chars / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Rule 3: No more than 30% of lines end with an ellipsis.
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False

    # Rule 4: At least 80% of words must contain at least one alphabetic character.
    alpha_words = sum(1 for word in words if re.search(r'[A-Za-z]', word))
    if (alpha_words / num_words) < 0.8:
        return False

    return True

def train_fasttext_model(dataset_path: str | Path, model_path: str | Path, validation_path: str | Path | None = None):
    """
    Train a fastText classifier model on the given labeled dataset.
    
    Args:
        dataset_path (str or Path): Path to the training data file.
        model_path (str or Path): Path to save the trained model.
        validation_path (str or Path, optional): Path to a validation dataset file.
    
    Returns:
        The trained fastText model.
    """
    model = train_supervised(input=str(dataset_path), epoch=50)
    model.save_model(str(model_path))
    
    if validation_path is not None:
        samples, precision, recall = model.test(str(validation_path))
        print(f'Validation -> Precision: {precision}, Recall: {recall}')
    
    print(f"Model saved to: {model_path}")
    return model

class QualityModel:
    def __init__(self, model_path: str | Path = 'models/fasttext-quality.bin'):
        self.model = load_model(str(model_path))
    
    def predict(self, text: str):
        label, prob = self.model.predict(text.replace('\n', ' '))
        # Remove the __label__ prefix.
        label = label[0].replace('__label__', '')
        if label == 'high':
            label = 'wiki'
        else:
            label = 'cc'
        return label, prob[0]
        
def load_and_predict(text: str):
    model = QualityModel('models/fasttext-quality.bin')
    return model.predict(text)

if __name__ == '__main__':
    # Define file paths.
    training_dataset = "data/quality-dataset-train.txt"
        
    train_fasttext_model(dataset_path=training_dataset, model_path="models/fasttext-quality.bin")
    
    sample_text = "This is a well-written research article with clear arguments and proper citations."
    label, confidence = load_and_predict(sample_text)
    print(f"Sample prediction -> Label: {label}, Confidence: {confidence}")

