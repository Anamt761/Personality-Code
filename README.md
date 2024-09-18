**Personality Trait Detection: Openness**

**Overview**

This project focuses on detecting the personality trait "Openness" using a dataset derived from the Myers-Briggs Type Indicator (MBTI) available on Kaggle. 
The primary goal is to classify text data according to the "Openness" trait, leveraging natural language processing (NLP) techniques and machine learning algorithms.

**Dataset**

**Name:** MBTI Type Dataset

**Available Link:** https://www.kaggle.com/datasets/datasnaek/mbti-type

**Description**: The dataset contains posts from individuals along with their respective MBTI personality types. The MBTI types are composed of four dichotomies: Introversion (I) vs. Extraversion (E), Intuition (N) vs. Sensing (S), Thinking (T) vs. Feeling (F), and Judging (J) vs. Perceiving (P).

**Background:**

The Myers–Briggs Type Indicator (MBTI) is a kind of psychological classification about humans experience using four principal psychological functions, sensation, intuition, feeling, and thinking, constructed by Katharine Cook Briggs and her daughter Isabel Briggs Myers.

From scientific or psychological perspective it is based on the work done on cognitive functions by Carl Jung i.e. Jungian Typology. This was a model of 8 distinct functions, thought processes or ways of thinking that were suggested to be present in the mind. Later this work was transformed into several different personality systems to make it more accessible, the most popular of which is of course the MBTI.

Recently, its use/validity has come into question because of unreliability in experiments surrounding it, among other reasons. But it is still clung to as being a very useful tool in a lot of areas, and the purpose of this dataset is to help see if any patterns can be detected in specific types and their style of writing, which overall explores the validity of the test in analysing, predicting or categorising behaviour.

**Focus:** This project focuses specifically on the "Openness" aspect, inferred from the Intuition (N) vs. Sensing (S) dimension.

**Setup Instruction:**  To run the code for this project, we use Google Colab GPU and Kaggle GPU for faster computation. The following libraries are required: numpy, spacy, pandas, matplotlib, nltk, and sklearn. These libraries and tools provide the necessary computational resources and functions for text processing, model training, and evaluation.

**Running the Code:**
Here is brief analysis of how to run the scripts or notebooks, including data preprocessing steps, training the models, and evaluating the results.


1. **Preprocessing:**
   - The preprocessing steps include the following:
     - Removal of URLs, emojis, digits, "|||", punctuation, and stopwords from the text data.
     - Lemmatization is applied to reduce words to their base forms.
  
2. **Features Applied:**
   - For **Shallow Machine Learning** and **Ensemble Models**, the following text features are used:
     - **TF-IDF** and **POS tagging** are integrated with models like Support Vector Machine (SVM), Decision Tree (DT), Logistic Regression (LR), Naive Bayes (NB), K-Nearest Neighbors (KNN), Random Forest (RF), XGBoost (XGB), Gradient Boosting (GB), and AdaBoost.
   - For **Deep Learning Models**, the following word embeddings are used:
     - **Word2Vec**, **GloVe**, and **Sentence Embeddings** are applied with deep models such as Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), and Transformer-based models like BERT.

3. **Performance Evaluation Measures:**
   - The models are evaluated based on the following metrics:
     - **Accuracy**, **Precision**, **Recall**, and **F1-score** to provide a comprehensive assessment of the model’s performance.

These steps ensure that the data is properly prepared, features are extracted, and appropriate models are trained and evaluated.

