import flask
import os
import pickle
import pandas as pd
import skimage
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

stopwords = stopwords.words('english')

app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = "models/vectorizer_ngram3.pkl"
path_to_text_classifier = "models/comments_model_ngram3.pkl"

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)


#####  pipeline from jupyter notebook ######
translator = str.maketrans('', '', string.punctuation)

def remove_stopwords(a):
    return " ".join([word for word in nltk.word_tokenize(a) if word not in stopwords])

def remove_sp_char(a):
    return a.translate(translator)

def text_pipeline2(a):

    a = remove_sp_char(a.lower())
    a = remove_stopwords(a)
    return a
###################################################

@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']
   
        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])
        
        # Make a prediction 
        predictions = model.predict(X)
        
        # Get the first and only value of the prediction.
        prediction = predictions[0]

        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)

        # Get the value of the first, and only, predicted proba.
        predicted_proba = predicted_probas[0]

        # The first element in the predicted probabs is the good posts
        bad_comment = predicted_proba[0]

        # The second element in predicted probas is % republican
        good_comment = predicted_proba[1]

        if bad_comment > 0.65:
            bad_comment *= 100;
            bad_comment= round(bad_comment, 2)
            good_comment *= 100;
            good_comment = round(good_comment, 2)
        elif good_comment > 0.65:
            bad_comment *= 100;
            bad_comment = round(bad_comment, 2)
            good_comment *= 100;
            good_comment = round(good_comment, 2)
        else:
            bad_comment = "Undecided"
            good_comment = "Undecided"


        return flask.render_template('main.html', 
            input_text=user_input_text,
            result=prediction,
            bad_percent=bad_comment,
            good_percent=good_comment)


@app.route('/bootstrap_twitter/')
def bootstrap():
    return flask.render_template('bootstrap_twitter.html')

@app.route('/about_us/')
def about_us():
    return flask.render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True)