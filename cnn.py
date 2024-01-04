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
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer

stop_words = stopwords.words("arabic")
# Load the dataset
data = pd.read_excel('Merge.xlsx')
print("data is read")
#from sklearn.preprocessing import LabelEncoder

# Separate positive and negative reviews
#positive_data = data[data['label']=="positive"]  # Assuming 4 and 5 are positive ratings
#negative_data = data[data['label']=="negative"]  # Assuming 0, 1, and 3 are negative ratings
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
        processed=' '.join(processed)
        processed_texts.append(processed)
        #print(processed)
        #print("******\n")
     # Lemmatization with ISRIStemmer
    stemmer = ISRIStemmer()
    lemmatized_texts = []
    #for tokens in processed_texts:
    #    lemmatized = [stemmer.stem(token) for token in tokens]
    #    lemmatized=' '.join(lemmatized)
    #    print(lemmatized)
    #    print("lem**ed*****")
    #    lemmatized_texts.append(lemmatized)
    print("without lemma :",processed_texts[0])
    #print("with lemma :",lemmatized_texts[0])
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    vectorized_data = vectorizer.fit_transform(processed_texts)
    
    return vectorized_data




# Assuming your data is stored in a pandas DataFrame called 'data'
data['label'] = data['label'].apply(lambda x: 1 if 'positive' in x else 0)

labels=data['label']


vectorized_data = preprocess_and_vectorize(texts)
features=vectorized_data

#for Gpu

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("CPU  ...!!!\n")
    
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

print("Features shape : ",X_train.shape)
print("Tst Features shape : ",X_test.shape)
print("Labels shape : ",y_train.shape)
print("Tst labels shape : ",y_test.shape)

# Reshape your input data to fit the model
X_train = X_train.todense().reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.todense().reshape((X_test.shape[0], X_test.shape[1], 1))

#current model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense

# Define the neural network model
vocab_size = 17000  # Vocabulary size
max_len = 40  # Maximum length of input sequences
embedding_dim = 300  # Embedding dimension

model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

# 1st Convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))

# 2nd Convolutional layer
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))

# Global Average Pooling
model.add(GlobalAveragePooling1D())

# Fully connected layer
model.add(Dense(units=8, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()

print("start model ")
# Train the model
#model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
#loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions
#predictions = model.predict(y_test)



# Save the model
model.save('my_model_cnn.h5')

# Save metrics to an external file
np.savetxt('training_loss_cnn.txt', history.history['loss'])
np.savetxt('training_accuracy_cnn.txt', history.history['accuracy'])
np.savetxt('val_loss_cnn.txt', history.history['val_loss'])
np.savetxt('val_accuracy_cnn.txt', history.history['val_accuracy'])

