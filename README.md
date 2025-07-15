# email_spam_classifier

📧 Email Scam Classifier

This project is a machine learning-based email scam detection system implemented in a Jupyter notebook. It classifies emails as **scam** or **non-scam** using natural language processing and classification algorithms.

🚀 Project Features

- Preprocessing of email text data (stopword removal, tokenization, etc.)
- Feature extraction using TF-IDF
- Classification using models like Logistic Regression, Naive Bayes, etc.
- Evaluation using accuracy, confusion matrix, and classification report

📂 Files

`email_scam_classifier.ipynb` – Main notebook containing all code and analysis

📌 Requirements

Make sure you have Python 3.x and the following libraries installed:

bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk


Also, for NLTK:

python
import nltk
nltk.download('stopwords')
nltk.download('punkt')


🧠 Model Workflow

1. Data Loading
2. Text Preprocessing:

   Lowercasing
   Removing stopwords
   Tokenization
3. Feature Engineering:

   TF-IDF Vectorization
4. Model Training:

   Logistic Regression / Naive Bayes
5. Evaluation:

   Accuracy, Precision, Recall, F1-score

📊 Sample Output
Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.96      0.95       149
           1       0.99      0.99      0.99       966
    accuracy                           0.99      1115
    macro avg      0.97      0.98      0.97      1115
    weighted avg   0.99      0.99      0.99      1115
