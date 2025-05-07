# use Google Colab or Jupyter

from datasets import load_dataset

dataset = load_dataset("Muennighoff/python-bugs")

# Access the dataset (it's a DatasetDict with a 'train' split)
train_dataset = dataset['train']

# Convert to Pandas DataFrame (if needed)
import pandas as pd
df = train_dataset.to_pandas()

# Save to CSV (or other formats)
df.to_csv("python_bugs.csv", index=False)
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input
from sklearn.preprocessing import LabelEncoder
# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# 1. Load the Dataset
try:
    df = pd.read_csv("/content/python_bugs.csv")
except FileNotFoundError:
    print("Error: CSV file not found. Please upload 'python_bugs.csv'.")
    exit()
    # 2. Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    else:
        return ''

df['prompt_clean'] = df['prompt'].apply(preprocess_text)
df['full_clean'] = df['full'].apply(preprocess_text)
# 3. TF-IDF for Bug Reports
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['prompt_clean']).toarray()

# 4. Word2Vec for Source Code
sentences = [text.split() for text in df['full_clean'] if isinstance(text, str)]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# 5. Word2Vec Embedding for Source Code
def get_word_vectors(text):
    if isinstance(text, str):
        words = text.split()
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
    return np.zeros(100)

word2vec_vectors = np.array([get_word_vectors(text) for text in df['full_clean']])
# 6. Combine TF-IDF and Word2Vec Features
X = np.concatenate((tfidf_matrix, word2vec_vectors), axis=1)
y = df['full']  # Using 'full' as a proxy for file location

# 7. Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# 9. CNN Model with Different Activation Functions
activation_functions = ['relu', LeakyReLU(negative_slope=0.1), 'sigmoid']

for activation in activation_functions:
    print(f"\nTraining model with {activation} activation...")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation=activation),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, verbose=0)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy ({activation}): {test_accuracy}")
    print(f"Test Loss ({activation}): {test_loss}")
    # 10. Bug Localization Function
def localize_bug(bug_description, source_code, model=None):
    if model is None:
        print("Please train a model before using the localize_bug function.")
        return None

    clean_bug_desc = preprocess_text(bug_description)
    tfidf_vector = tfidf_vectorizer.transform([clean_bug_desc]).toarray()

    clean_source_code = preprocess_text(source_code)
    source_vector = get_word_vectors(clean_source_code)

    combined_features = np.concatenate((tfidf_vector, source_vector.reshape(1, -1)), axis=1)
    prediction = model.predict(combined_features, verbose=0)
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    try:

        buggy_line = df[df['full'] == predicted_label]['buggy_line'].values[0]

        code_lines = df[df['full'] == predicted_label]['full'].values[0].split('\n')
        buggy_code_snippet = code_lines[buggy_line - 1]  # Extract buggy line

        formatted_code_snippet = ""
        for i, line in enumerate(code_lines):
            line_number = i + 1
            formatted_code_snippet += f"{line_number}: {line}\n"

    except KeyError:
        buggy_line = "N/A"
        formatted_code_snippet = "Buggy line information not available."

    # Simplified output format (updated)
    output = f"""
Predicted_File_Location: {predicted_label}


Explanation how to solve the BUG:
The model identified a potential bug in the code snippet. Please review the code and make the necessary corrections.
"""

    return output
# Train with the best activation
best_activation = 'relu'
print(f"\nTraining final model with {best_activation} activation...")
final_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation=best_activation),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, verbose=0)

new_bug_description = "Add a check for null values in the data."
new_source_code = "def process_data(data):\n    return data.dropna()"
predicted_location = localize_bug(new_bug_description, new_source_code, model=final_model)
if predicted_location:
    print(predicted_location)
    import time

def calculate_time_complexity():
    start_time = time.time()

    # 1. Data Loading
    loading_start = time.time()
    try:
        df = pd.read_csv("/content/python_bugs.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Please upload 'python_bugs.csv'.")
        exit()
    loading_time = time.time() - loading_start
    print(f"Data Loading Time: {loading_time:.4f} seconds")

    # 2. Text Preprocessing
    preprocessing_start = time.time()
    df['prompt_clean'] = df['prompt'].apply(preprocess_text)
    df['full_clean'] = df['full'].apply(preprocess_text)
    preprocessing_time = time.time() - preprocessing_start
    print(f"Text Preprocessing Time: {preprocessing_time:.4f} seconds")

    # 3. TF-IDF Vectorization
    tfidf_start = time.time()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['prompt_clean']).toarray()
    tfidf_time = time.time() - tfidf_start
    print(f"TF-IDF Vectorization Time: {tfidf_time:.4f} seconds")

    # 4. Word2Vec Training
    word2vec_start = time.time()
    sentences = [text.split() for text in df['full_clean'] if isinstance(text, str)]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_time = time.time() - word2vec_start
    print(f"Word2Vec Training Time: {word2vec_time:.4f} seconds")

    # 5. Word2Vec Embedding
    embedding_start = time.time()
    word2vec_vectors = np.array([get_word_vectors(text) for text in df['full_clean']])
    embedding_time = time.time() - embedding_start
    print(f"Word2Vec Embedding Time: {embedding_time:.4f} seconds")

    # 6. Feature Combination
    combine_start = time.time()
    X = np.concatenate((tfidf_matrix, word2vec_vectors), axis=1)
    combine_time = time.time() - combine_start
    print(f"Feature Combination Time: {combine_time:.4f} seconds")

    # 7. Label Encoding
    encoding_start = time.time()
    y_encoded = label_encoder.fit_transform(y)
    encoding_time = time.time() - encoding_start
    print(f"Label Encoding Time: {encoding_time:.4f} seconds")

    # 8. Train-Test Split
    split_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    split_time = time.time() - split_start
    print(f"Train-Test Split Time: {split_time:.4f} seconds")

    # 9. Model Training
    training_start = time.time()
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, verbose=0)
    training_time = time.time() - training_start
    print(f"Model Training Time: {training_time:.4f} seconds")

    # 10. Model Evaluation
    evaluation_start = time.time()
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    evaluation_time = time.time() - evaluation_start
    print(f"Model Evaluation Time: {evaluation_time:.4f} seconds")

    # Total Time
    total_time = time.time() - start_time
    print(f"Total Execution Time: {total_time:.4f} seconds")

calculate_time_complexity()
 0