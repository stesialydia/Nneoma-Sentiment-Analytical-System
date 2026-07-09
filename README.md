**Sentiment Analytical System**

**Overview**
The Sentiment Analytical System is a machine learning project designed to analyse patient feedback on prescribed medication and classify sentiment to support improved patient satisfaction. The system helps identify whether patient reviews express positive or negative experiences, allowing healthcare teams, analysts, or decision-makers to better understand medication-related concerns and patient perceptions.

This project compares two text classification models:

Random Forest Classifier
Naive Bayes Classifier

After model evaluation, the Naive Bayes classifier was selected as the final model because it delivered better performance for the sentiment classification task. The trained model and vectorizer were saved using Pickle and deployed through a Flask web application.

**Project Objective**

The main objective of this project is to build an NLP-based sentiment analysis system that can:

#Analyse patient medication reviews.
#Convert text reviews into machine-readable features.
#Classify sentiment as positive or negative.
#Compare the performance of Random Forest and Naive Bayes models.
#Deploy the best-performing model in a simple Flask application.
#Support data-driven understanding of patient satisfaction with medication.

**Key Features**
Text cleaning and preprocessing
Tokenization
Stop-word removal
Lemmatization
Sentiment label conversion
TF-IDF vectorization
Class imbalance handling
Model comparison between Random Forest and Naive Bayes
Model evaluation using classification metrics
Model and vectorizer serialization with Pickle
Flask web app for sentiment prediction

**Tech Stack**
Python
Pandas
NumPy
Scikit-learn
NLP
NLTK
Flask
Pickle
HTML/CSS

**Machine Learning Workflow**
The project follows a complete NLP machine learning pipeline:

Load the medication review dataset.
Clean and preprocess text data.
Tokenize text reviews.
Remove stop words.
Apply lemmatization.
Convert sentiment ratings into binary labels.
Transform text into numerical features using TF-IDF vectorization.
Split the dataset into training and testing sets.
Check class balance and apply sampling where necessary.
Train Random Forest and Naive Bayes classifiers.
Evaluate model performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.
Tune model parameters where required.
Select the best-performing model.
Save the trained model and vectorizer using Pickle.
Deploy the model using Flask.

**Model Selection**

Two models were compared during development:

**Random Forest Classifier**
Random Forest was tested because it is a strong ensemble learning algorithm that can perform well on many classification problems. It was used as a benchmark model for comparing performance on the medication sentiment dataset.

**Naive Bayes Classifier**
Naive Bayes was selected as the final model because it performed better for this text classification task. Naive Bayes is commonly effective for NLP problems because it works well with high-dimensional sparse text features produced by vectorizers such as TF-IDF.

**Why Naive Bayes Was Chosen**
Naive Bayes was chosen as the final model because it provided stronger performance during evaluation. It was also lightweight, efficient, and well-suited for sentiment classification using TF-IDF features.

**The final pipeline uses:**

TF-IDF vectorizer for feature extraction
Naive Bayes classifier for sentiment prediction
Pickle for saving the trained model and vectorizer
Flask for deployment

**Evaluation Metrics**

The models were evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion matrix
ROC curve

These metrics helped compare the Random Forest and Naive Bayes models and supported the final model selection.

**Project Structure**

sentiment-analytical-system/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── templates/
│   └── index.html
    └── result.html
├── static/images
│   └── background.jpg
├── Procfile
├── request.py
├── model.py
├── .gitignore
├── runtime.txt
├── requirements.txt
└── README.md

**Note:** File and folder names may differ depending on your local project structure.

**Installation**

Clone the repository:
git clone https://github.com/your-username/sentiment-analytical-system.git
cd sentiment-analytical-system

Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Running the Flask App
Start the Flask application:
python app.py
Open your browser and go to:
http://127.0.0.1:5000/

Enter a medication review into the input field and submit it to receive a sentiment prediction.

**Example Use Case**
A patient writes a review about their experience with a prescribed medication. The system preprocesses the review, converts it into TF-IDF features, and uses the trained Naive Bayes model to classify the sentiment.
This can help identify patterns in patient feedback, highlight medication-related concerns, and support better patient satisfaction analysis.

**Model Saving**

The final Naive Bayes model and vectorizer were saved using Pickle:

import pickle

with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    
These files are loaded in the Flask app to make predictions on new patient reviews.

**Future Improvements**
Add more medication review data for improved generalization.
Deploy the app to a cloud platform.
Add probability scores for model confidence.
Include neutral sentiment classification.
Improve the user interface of the Flask app.
Add model explainability for predicted sentiment.
Track sentiment trends across medication categories.

**Conclusion**
This project demonstrates how natural language processing and machine learning can be used to analyse patient medication reviews and support patient satisfaction insights. By comparing Random Forest and Naive Bayes classifiers, the project selected the best-performing model and deployed it through a Flask application for practical use.

**Author**
Nneoma Nnorom
AI and Data Science Engineer
