#!/usr/bin/env python
# coding: utf-8

# # importing packages and libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Conv1D, GlobalMaxPooling1D, Dense
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import tensorflow as tf





# # importing Datasets

# In[2]:


df1 = pd.read_csv("C:\\Users\\bhanu\\OneDrive\\Desktop\\NLP\\True (1).csv", encoding='latin1',
                  on_bad_lines='skip') 
df1


# In[3]:


df2 = pd.read_csv("C:\\Users\\bhanu\\OneDrive\\Desktop\\NLP\\Fake (1).csv",encoding='latin-1',on_bad_lines='skip')  
df2


# In[4]:


# Concatenate the datasets
df3 = pd.concat([df1, df2], ignore_index=True)
df3


# # EDA 

# In[5]:


df3.head()


# In[6]:


df3.tail()


# In[7]:


df3.shape


# In[8]:


df3.info()


# In[9]:


df3.describe()


# In[10]:


df3.isnull()


# In[11]:


missing_values = df3.isnull().sum()
print("Missing values:\n", missing_values)


# In[12]:


df3.fillna("<MISSING>", inplace=True)


# In[13]:


missing_values = df3.isnull().sum()
print("Missing values:\n", missing_values)


# In[14]:


duplicate_records = df3[df3.duplicated()]
print("Duplicate records:", duplicate_records)


# In[15]:


df3.drop_duplicates(inplace=True)


# In[16]:


text=df3["text"]
text


# # applying tokenization

# In[17]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


# In[18]:


# Apply the tokenization function to the 'text' column
df3['tokenized_text'] = df3['text'].apply(tokenize_text)

df3['tokenized_text']


# In[19]:


# Display the tokenized text for the first few rows
for index, row in df3.head().iterrows():
    print(f"Row {index}:")
    print(row['tokenized_text'])


# # stopwords

# In[20]:


from nltk.corpus import stopwords

# Example: Remove English stopwords
stopwords_list = set(stopwords.words('english'))
stopwords_list


# In[21]:


df3['filtered_tokens']= df3['tokenized_text'].apply(lambda tokens: [token for token in tokens if token.lower() not in stopwords_list])


# In[22]:


df3['filtered_tokens']


# In[23]:


# Define additional custom stopwords
custom_stopwords = {'WASHINGTON', 'Reuters', 'head', 'conservation','SEATTLE/WASHINGTON','Transgender','special','Trump','campaign','21st','Century',' Wire','says','21WIRE','reported','familiar',' theme','predicted','Henningsen','said','Said','US'}


# In[24]:


# Add custom stopwords to the set
stopwords_list.update(custom_stopwords)


# In[25]:


stopwords_list


# # Tokenization, Normalization, and Lemmatization

# In[25]:


# Tokenization, Normalization, and Lemmatization
lemmatizer = WordNetLemmatizer()


# In[26]:


def process_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text
    # Tokenization
    tokens = word_tokenize(text)
    tokens
    # Normalization
    tokens = [token.lower() for token in tokens]
    tokens
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

df3['processed_text'] = df3['text'].apply(process_text)

# Remove stopwords
df3['filtered_tokens'] = df3['processed_text'].apply(lambda tokens: [token for token in tokens if token.lower() not in stopwords_list])


# # Data Visualization

# In[ ]:


# the VADER sentiment analyzer


# In[27]:


# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each text in the DataFrame
df3['sentiment'] = df3['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Classify sentiment into categories (positive, neutral, negative)
df3['sentiment_category'] = df3['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
df3['sentiment_category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# word cloud


# In[29]:


# Data Visualization - Word Cloud
text_data = " ".join(df3['text'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()


# # Data modelling

# # CNN (Convolutional neural network)

# In[30]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df3['filtered_tokens'], df3['subject'], test_size=0.2, random_state=42)


# In[31]:


# Label encoding for training set
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Check for unseen labels in test set
unseen_labels = set(y_test) - set(label_encoder.classes_)

if unseen_labels:
    print("Unseen labels in test set:", unseen_labels)
    # Filter out unseen labels from test set
    mask = y_test.isin(label_encoder.classes_)
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
else:
    X_test_filtered = X_test
    y_test_filtered = y_test

# Encode filtered test set labels
y_test_encoded = label_encoder.transform(y_test_filtered)

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test_filtered)
max_sequence_length = max([len(seq) for seq in X_train_seq + X_test_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')



# In[32]:


# Define CNN model
cnn_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=64, activation='relu'),
    Dense(units=len(label_encoder.classes_), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test_encoded))


# In[33]:


# Evaluate CNN model
cnn_pred = cnn_model.predict(X_test_pad)
cnn_acc = accuracy_score(y_test_encoded, np.argmax(cnn_pred, axis=1))
print("CNN Accuracy:", cnn_acc)
cnn_mse = mean_squared_error(y_test_encoded, np.argmax(cnn_pred, axis=1))
print("CNN Mean Squared Error:", cnn_mse)


# # Naive Bayes

# In[34]:


# Assuming you have a dataset with 'text' and 'label' columns
X = df3['text']
y = df3['sentiment_category']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer, TfidfTransformer, and MultinomialNB
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluation
print(classification_report(y_test, y_pred))


# # Random Forest classifier

# In[44]:


# Assuming you have a dataset with 'text' and 'sentiment_category' columns
# Split the dataset into features (X) and target labels (y)
X = df3['text']
y = df3['sentiment_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # Logistic Regression

# In[47]:


# Assuming you have a dataset with 'text' and 'sentiment_category' columns
# Split the dataset into features (X) and target labels (y)
X = df3['text']
y = df3['sentiment_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the classifier (Logistic Regression in this case)
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # LSTM (Long short team memory)

# In[58]:


# Convert numpy arrays to lists of strings and ensure all elements are strings
X_train_texts = [str(text) for text in X_train.tolist()]
X_test_texts = [str(text) for text in X_test.tolist()]

# Tokenize and pad sequences for LSTM
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train_texts)
X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define LSTM model
lstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
lstm_model.fit(X_train_pad, y_train_encoded, epochs=3, batch_size=64, validation_data=(X_test_pad, y_test_encoded))

# Evaluate the LSTM model
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
accuracy_lstm = accuracy_score(y_test_encoded, y_pred_lstm)
print("LSTM Accuracy:", accuracy_lstm)
print("LSTM Classification Report:")
print(classification_report(y_test_encoded, y_pred_lstm))


# In[28]:


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Drop rows with missing values (NaNs) in the 'text' or 'label' columns
df3 = df3.dropna(subset=['text', 'sentiment_category'])

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df3['text'])
y = df3['sentiment_category']  # Assuming 'label' is your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')  # You can try different kernels like 'rbf' or 'sigmoid'

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# # from the above Data modelling methods we can see that the Random Forest classifier is driving the high accuracy

# In[ ]:





# In[ ]:




