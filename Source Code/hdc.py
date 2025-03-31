
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import random
import string

# Load the dataset
df = pd.read_csv('/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only label and text columns
df.columns = ['label', 'text']

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'],
                                                   test_size=0.2, random_state=42)

class HDCSpamDetector:
    def __init__(self, D=10000, ngram_range=(1, 3)):
        """
        Initialize HDC spam detector

        Parameters:
        - D: Dimension of hypervectors
        - ngram_range: Range of n-grams to consider
        """
        self.D = D
        self.ngram_range = ngram_range
        self.item_memory = {}  # Stores hypervectors for ngrams
        self.class_hypervectors = {}  # Stores class prototype hypervectors
        self.vectorizer = None

    def generate_random_hypervector(self):
        """Generate a random binary hypervector"""
        return np.random.randint(0, 2, self.D)

    def bind(self, hv1, hv2):
        """Binding operation (XOR)"""
        return np.bitwise_xor(hv1, hv2)

    def bundle(self, hypervectors):
        """Bundling operation (majority vote)"""
        return (np.sum(hypervectors, axis=0) > len(hypervectors)/2).astype(int)

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        return text

    def build_item_memory(self, texts):
        """Build the item memory (ngram to hypervector mapping)"""
        # Initialize vectorizer to extract ngrams
        self.vectorizer = CountVectorizer(analyzer='char',
                                        ngram_range=self.ngram_range,
                                        min_df=1)
        self.vectorizer.fit(texts)

        # Generate random hypervectors for each ngram
        all_ngrams = self.vectorizer.get_feature_names_out()
        for ngram in all_ngrams:
            self.item_memory[ngram] = self.generate_random_hypervector()

    def text_to_hypervector(self, text):
        """Convert text to a single hypervector"""
        text = self.preprocess_text(text)
        ngrams = self.vectorizer.build_analyzer()(text)

        if not ngrams:
            return np.zeros(self.D)

        # Get hypervectors for each ngram
        ngram_hvs = [self.item_memory[ngram] for ngram in ngrams
                    if ngram in self.item_memory]

        if not ngram_hvs:
            return np.zeros(self.D)

        # Bundle all ngram hypervectors
        return self.bundle(ngram_hvs)

    def train(self, X_train, y_train):
        """Train the HDC model"""
        # Build item memory
        self.build_item_memory(X_train)

        # Convert each text to hypervector
        train_hvs = []
        for text in X_train:
            train_hvs.append(self.text_to_hypervector(text))
        train_hvs = np.array(train_hvs)

        # Create class prototype hypervectors
        for label in [0, 1]:
            class_samples = train_hvs[y_train == label]
            if len(class_samples) > 0:
                self.class_hypervectors[label] = self.bundle(class_samples)
            else:
                self.class_hypervectors[label] = np.zeros(self.D)

    def predict(self, text):
        """Predict class for a single text"""
        hv = self.text_to_hypervector(text)

        # Calculate Hamming distance to each class prototype
        distances = {}
        for label, class_hv in self.class_hypervectors.items():
            distances[label] = np.sum(np.bitwise_xor(hv, class_hv))

        # Return class with smallest distance
        return min(distances.items(), key=lambda x: x[1])[0]

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        y_pred = []
        for text in X_test:
            y_pred.append(self.predict(text))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return y_pred

# Initialize and train the HDC model
hdc_model = HDCSpamDetector(D=20000, ngram_range=(1, 3))
hdc_model.train(X_train, y_train)

# Evaluate on test set
y_pred = hdc_model.evaluate(X_test, y_test)

# Test with individual messages
test_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Hey, how are you doing today?",
    "URGENT! You have won a 1 week FREE membership",
    "Let's meet for lunch tomorrow"
]

for msg in test_messages:
    prediction = hdc_model.predict(msg)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {'spam' if prediction == 1 else 'ham'}")