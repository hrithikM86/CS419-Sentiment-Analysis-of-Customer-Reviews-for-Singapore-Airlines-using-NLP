# CS-419(M) - "I hated it. 8/10"

## Group 7

**Team Members:**
- Hrithik Mhatre
- Vineet Jaiswal
- Vaishnav Vernekar


---

## Project Introduction: "I hated it. 8/10"

This project aims to generate customer ratings (on a scale of 1 to 5) solely from their textual reviews. We focus on leveraging Natural Language Processing (NLP) and Machine Learning techniques to predict the missing ratings from customer reviews for a sample airline company. This project showcases how customer reviews alone can help reconstruct ratings, thus offering insights into customer satisfaction and enabling better decision-making for businesses.

### Objectives:
1. **Generate Customer Ratings from Text Reviews:** Reconstruct customer ratings (1-5) solely based on the textual reviews.
2. **NLP & Machine Learning Models:** Use machine learning and deep learning models to analyze text reviews and predict ratings.
3. **Business Value:** Help airlines mitigate oversight and improve customer satisfaction by understanding textual feedback.

### Additional Aim:
- **Word Embeddings Comparison:** Compare different word embedding methods and determine the best approach for predicting customer ratings.

## Word Embedding Methods:
1. **One Hot Encoding + Random Forest**
2. **Bag of Words + Random Forest**
3. **C-BOW + XGBoost**
4. **Pre-trained Word Embeddings (Word2Vec) + XGBoost / SVM / Neural Networks (NN)**

## Dataset

We used the [Singapore Airlines Reviews Dataset](https://www.kaggle.com) containing around 10,000 anonymized customer reviews with corresponding ratings. The dataset provides valuable insights into customer perceptions and satisfaction levels regarding Singapore Airlines.

---

## Machine Learning Techniques in NLP

NLP techniques applied include:
- **Tokenization**: Breaking down text into tokens.
- **Parsing**: Analyzing sentence structures.
- **Stemming & Lemmatization**: Reducing words to their root/base forms.
- **Part-of-Speech Tagging**: Assigning grammatical categories to words.
- **Feature Extraction**: Extracting relevant features for Machine Learning models.

### Data Preprocessing:
1. **Data Cleaning:** Removing special characters, punctuation, numbers, and irrelevant information.
2. **Text Normalization:** Standardizing text (lowercase conversion, lemmatization).
3. **Stopword Removal:** Removing frequently occurring but insignificant words.
4. **Tokenization:** Breaking text into individual words or sentences for analysis.

---

## Training & Validation Datasets

We split the dataset into two parts:
- **Training Data:** 80% of the data
- **Validation Data:** 20% of the data  
We used `train_test_split()` with `test_size=0.2` to achieve this.

---

## Models and Results

### One Hot Encoding + Random Forest
- **Hyperparameters:** `n_estimators=100`, `criterion='entropy'`
- **Accuracy:** 62.77%
- **Challenges:** High dimensionality and sparse representation; lack of contextual information.

### Bag of Words + Random Forest
- **Max Features:** 500
- **Hyperparameters:** Same as above
- **Accuracy:** 62.77%
- **Advantages:** Frequency of words considered; still lacks contextual meaning.

### CBOW (Continuous Bag of Words) + XGBoost
- **Hyperparameters:**
  - `vocab_size=len(vocabulary)`
  - `embed_size=100`
  - `learning_rate=0.001`, `epochs=25`, `batch_size=256`
  - `n_estimators=100`, `criterion='entropy'`
- **Accuracy:** 53.50%
- **Challenges:** Computational restrictions impacted the model performance.

### Pre-Trained Word
