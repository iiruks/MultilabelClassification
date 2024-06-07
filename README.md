# Multi-Label Classification for Toxic Comment Detection
This Python script performs multi-label classification for toxic comment detection using various machine learning algorithms. It preprocesses the data, explores its characteristics, and trains several classifiers to predict the toxicity level of comments.

## Prerequisites
Make sure you have the following Python libraries installed:
os
csv
pandas
numpy
matplotlib
seaborn
IPython
Additionally, you will need to install the wordcloud, nltk, scikit-learn, and scikit-multilearn libraries. You can install them using pip:

## Usage
Clone this repository or download the script toxic_comment_classification.py.
Place your training dataset in the specified location (data_path variable).
Open the script in your Python environment (e.g., Jupyter Notebook) and execute it.

## Data Exploration
The script loads the dataset and provides insights into its structure and characteristics:
Number of rows and columns
Sample data
Missing values check
Distribution of comments across different categories
Visualization of comment counts in each category
Visualization of comments with multiple labels
WordCloud representation of most used words in each category

## Data Preprocessing
The data preprocessing steps include:
Cleaning HTML tags
Cleaning punctuations and special characters
Removing stop words
Stemming words
Train-test split
TF-IDF vectorization

## Multi-Label Classification
The script implements several multi-label classification techniques:
One Vs Rest Classifier
Binary Relevance
Classifier Chains
Label Powerset
Adapted Algorithm (MLkNN)
Each technique is applied to classify comments into different toxicity categories.
