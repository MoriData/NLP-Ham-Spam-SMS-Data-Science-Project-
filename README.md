# NLP Ham-Spam SMS Classification

This project demonstrates the use of Natural Language Processing (NLP) to classify SMS messages into two categories: *ham* (legitimate messages) and *spam* (unsolicited or junk messages). The project utilizes machine learning techniques to process text data, extract features, and train a model for SMS classification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Modeling Process](#modeling-process)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This project leverages a dataset containing labeled SMS messages categorized as *ham* (legitimate) or *spam*. The objective is to predict the class of new messages using machine learning algorithms. Various preprocessing steps are applied, such as text cleaning, tokenization, and vectorization. A classification model is then trained and evaluated for accuracy.

## Dataset

The dataset used in this project is publicly available on Kaggle and consists of SMS messages labeled as either *ham* or *spam*. Each message in the dataset contains text data and is categorized based on whether it's a legitimate message (*ham*) or a spam message (*spam*).

You can find the dataset at: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Libraries Used

- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `matplotlib`, `seaborn`: For data visualization
- `sklearn`: For machine learning algorithms and preprocessing tools
- `nltk`: For text processing and NLP tasks

## Modeling Process

1. **Data Cleaning**: 
   - Removed non-alphanumeric characters and stop words.
   - Normalized text (lowercased and removed punctuation).

2. **Feature Extraction**:
   - Used `TfidfVectorizer` to convert text data into numerical format suitable for machine learning algorithms.

3. **Modeling**:
   - Tried several classification models, including Logistic Regression, Support Vector Machines (SVM), and Naive Bayes.
   - Evaluated model performance based on accuracy and other relevant metrics.

4. **Model Evaluation**:
   - Used metrics such as precision, recall, and F1-score to evaluate model performance.
   - Performed cross-validation to ensure the model generalizes well on unseen data.

## Evaluation

- The models were evaluated based on accuracy and the ability to distinguish between spam and ham messages. 
- The most optimal model was selected after comparing results from different algorithms.

## Results

The final model achieved a high level of accuracy in classifying SMS messages as either *ham* or *spam*. The evaluation metrics can be found in the notebook.



