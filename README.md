# Disaster Tweet Analysis
This is an analysis for a Twitter text dataset to classify whether a post is about a real disaster.

0. References
   - This analysis is created referring to the works done by GUNES EVITAN and Alexia Audevart. The links are avairable at the first cell.

1. Preparations
   - Basic libararies: numpy, pandas, string, re, random, time, itertools
   - Data Profiling: ydata_profiling
   - Visualization libraries: matplotlib, seaborn
   - Data Cleaning: ftfy, geopy
   - NLP libraries: spacy, nltk
   - Models: sklearn, tensorflow, transformers

2. Dataset installation
   - CSV file name: '/kaggle/input/nlp-getting-started/train.csv' from Kaggle competition "Natural Language Processing with Disaster Tweets"
   - Profiling report
   - Create a combined dataset for convenience

3. Preprocessing
   - Text Cleaning
     - text_cleaning function:
       - Remove URls
       - Transform Unicodes using ftfy
       - Split punctuations from words to prevent generating new words consist of punctuation(s) and existing word(s)
       - Make all letters in lower cases for convenience
       - Map dictionaries with keywords of abbreviations and slangs that are likely to appear in Twitter texts and values of corresponding original texts
     - Generate a 'cleaned_text' column (without overriding any columns) in the combined dataset using the text_cleaning function
   - Data Transformation
     - lemmatize_and_remove_stopwords function:
       - Using spacy's loading object 'en_core_web_sm' to create tokens so that lemmatization and stopword removal can be performed
       - Create a 'transformed_text' column (without overriding any columns) in the combined dataset using the text_cleaning function
   - Keywords
     - Using the cleaned text to re-extract the keywords
     - Add newly extracted keywords to keyword column
   - Locations
     - Using Nominatim via geopy to extract the address of locations that appear in a text
     - Need a lot of time to run the code because Nominatim is a free resource and restrict users to access a location per second
   - Dataset Rearrangement
     - Generate a json file from dataset combined
     - Read the file and split it back into train and test datasets
     - This reduces a significant amount of time by avoiding preprocessing (debugging purpose)

4. EDA
   - Text
     - Numerical features
       - The length of text
       - The number of words
       - The average length of words
       - The number of stopwords
       - The number of hashtags (#)
       - The number of mentions (@)
       - The number of punctuations
       - The number of urls
     - Hashtag content
       - Specific hashtag labels can be helpful to classify the text
       - Make a score of the hashtag labels each text contains
     - Sentiment analysis
       - The sentiment of a text may relate to the target
       - Use SentimentIntensityAnalyzer
       - Take the compound score for feature
     - Location, revisited
       - By reviewing the address column extracted from location, notice they are not as specific as expected, so take country information and make 'country' column

5. Modeling
   - DistilBERT
     - Combine the text columns with 'keyword', '#_content', and 'country' columns to generate 'final_text' column that is used to perform distilBERT modeling using the [CLS] and [SEP] tokens.
     - Imprement distilBERT modeling with 'distilbert-base-uncased'
     - Select numerical features from the datasets and normalize them
   - Custom model
     - Create a custom model by setting input, dense, and output
     - Fit it with 100 epochs with early stopping of patience 20 to see what epochs are balanced for this model
     - Adopt 30 epochs and refit the model for predictions
   - Scikit-learn model
     - Combine the features obtained by distilBERT modeling and numerical features to create the input dataset
     - Hyperparameter tuning
       - Using GridSearchCV to find the best model and its parameter set
       - LogisticRegression with L1 penaralization and parameters C=1.0, solver='saga' is the best
       - Visualize it with confusion matrices
6. Model Selection
   - Select either custom model or scikit-learn model to submit
   - Surprisingly, scikit-learn model performs better in the train dataset
    
