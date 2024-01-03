from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from pyarabic.araby import strip_tashkeel, strip_tatweel, normalize_ligature, normalize_hamza, normalize_alef, normalize_teh
#from pyarabic.araby import ArabicStopWords
from pyarabic.araby import is_arabicrange
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
import pandas as pd
from pyarabic.araby import strip_tashkeel, strip_tatweel, normalize_ligature, normalize_hamza, normalize_alef, normalize_teh


stop_words = stopwords.words("arabic")
# Load the dataset
data = pd.read_excel('egy_tweet.xlsx')
#from sklearn.preprocessing import LabelEncoder

# Separate positive and negative reviews
positive_data = data[data['label']=="positive"]  # Assuming 4 and 5 are positive ratings
negative_data = data[data['label']=="negative"]  # Assuming 0, 1, and 3 are negative ratings
# Preprocess the data - Assuming the 'text' column contains the tweet text
texts = data['review'].values
labels = data['label'].values  # Assuming there's a 'label' column indicating sentiment or some target




nltk.download('punkt')


#preprocessing:

def preprocess_and_vectorize(X):
    # Tokenization with PyArabic
    tokenized_texts = []
    for text in X:
        tokens = [word for word in text.split() if is_arabicrange(word)]
        tokenized_texts.append(tokens)
    
    # Lemmatization with PyArabic (PyArabic does not have built-in lemmatization)
    # As PyArabic does not have a built-in lemmatizer, you can skip this step or use other libraries
    
    # Normalize the text using PyArabic
    normalized_texts = []
    for tokens in tokenized_texts:
        normalized = [normalize_ligature(token) for token in tokens]
        normalized = [normalize_hamza(token) for token in normalized]
        normalized = [normalize_alef(token) for token in normalized]
        normalized = [normalize_teh(token) for token in normalized]
        normalized_texts.append(normalized)
    
    # Remove Arabic stopwords
    arabic_stopwords =arabic_stopwords = set(stopwords.words('arabic'))
    processed_texts = []
    for text in normalized_texts:
        processed = [token for token in text if token not in arabic_stopwords]
        processed_texts.append(processed)
        #print(processed)
        #print("******\n")
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    vectorized_data = vectorizer.fit_transform(processed_texts)
    
    return vectorized_data




# Assuming your data is stored in a pandas DataFrame called 'data'
data['label'] = data['label'].apply(lambda x: 1 if 'positive' in x else 0)

labels=data['label']


vectorized_data = preprocess_and_vectorize(texts)
features=vectorized_data



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Reshape your input data to fit the model
X_train = X_train.todense().reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.todense().reshape((X_test.shape[0], X_test.shape[1], 1))
X_train = X_train.reshape((32000, 52350, 1))


#current model

# Define the model architecture
embedding_dim=100
vocab_size=52350

model = Sequential()

model.add(Embedding(input_dim=vocab_size,  # Replace vocab_size with your actual vocabulary size
                    output_dim=embedding_dim,  # Choose an appropriate embedding dimension
                    input_length=X_train.shape[1]))
model.add(Dropout(0.5))  # Apply dropout after the embedding layer

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=64))
model.add(Dropout(0.5))  # Apply dropout before the final dense layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions
#predictions = model.predict(y_test)



# Save the model
model.save('my_model_cnnLSTM1.h5')

# Save metrics to an external file
np.savetxt('training_loss.txt', history.history['loss'])
np.savetxt('training_accuracy.txt', history.history['accuracy'])
np.savetxt('val_loss.txt', history.history['val_loss'])
np.savetxt('val_accuracy.txt', history.history['val_accuracy'])

