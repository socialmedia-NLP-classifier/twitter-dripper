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

# path_to_vectorizer = 'models/vectorizer.pkl'
# path_to_text_classifier = 'models/text-classifier.pkl'
# path_to_image_classifier = 'models/image-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

# with open(path_to_image_classifier, 'rb') as f:
#     image_classifier = pickle.load(f)



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
        # Just render the initial form, to get input
        return(flask.render_template('main_twitter.html'))
    
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
            bad_comment = "Yes"
            good_comment = "No"
        elif good_comment > 0.65:
            bad_comment = "No"
            good_comment = "Yes"
        else:
            bad_comment = "Undecided"
            good_comment = "Undecided"
        
        return flask.render_template('main_twitter.html', 
            input_text=user_input_text,
            result=prediction,
            bad_percent=bad_comment,
            good_percent=good_comment)


# @app.route('/input_values_twitter/', methods=['GET', 'POST'])
# def input_values():
#     if flask.request.method == 'GET':
#         # Just render the initial form, to get input
#         return(flask.render_template('input_values_twitter.html'))

#     if flask.request.method == 'POST':
#         # Get the input from the user.
#         var_one = flask.request.form['input_variable_one']
#         var_two = flask.request.form['another-input-variable']
#         var_three = flask.request.form['third-input-variable']

#         list_of_inputs = [var_one, var_two, var_three]

#         return(flask.render_template('input_values.html', 
#             returned_var_one=var_one,
#             returned_var_two=var_two,
#             returned_var_three=var_three,
#             returned_list=list_of_inputs))

#     return(flask.render_template('input_values.html'))


# @app.route('/images_twitter/')
# def images():
#     return flask.render_template('images_twitter.html')


@app.route('/bootstrap_twitter/')
def bootstrap():
    
    return flask.render_template('bootstrap_twitter.html')

# def main():
#     if flask.request.method == 'GET':
#         # Just render the initial form, to get input
#         return(flask.render_template('main_twitter.html'))
    
#     if flask.request.method == 'POST':
#         # Get the input from the user.
#         user_input_text = flask.request.form['user_input_text']
        
#         # Turn the text into numbers using our vectorizer
#         X = vectorizer.transform([user_input_text])
        
#         # Make a prediction 
#         predictions = model.predict(X)
        
#         # Get the first and only value of the prediction.
#         prediction = predictions[0]

#         # Get the predicted probabs
#         predicted_probas = model.predict_proba(X)

#         # Get the value of the first, and only, predicted proba.
#         predicted_proba = predicted_probas[0]

#         # The first element in the predicted probabs is the good posts
#         bad_comment = predicted_proba[0]

#         # The second element in predicted probas is % republican
#         good_comment = predicted_proba[1]


#         return flask.render_template('main_twitter.html', 
#             input_text=user_input_text,
#             result=prediction,
#             bad_percent=bad_comment,
#             good_percent=good_comment)


# @app.route('/classify_image_twitter/', methods=['GET', 'POST'])
# def classify_image():
#     if flask.request.method == 'GET':
#         # Just render the initial form, to get input
#         return(flask.render_template('classify_image_twitter.html'))

#     if flask.request.method == 'POST':
#         # Get file object from user input.
#         file = flask.request.files['file']

#         if file:
#             # Read the image using skimage
#             img = skimage.io.imread(file)

#             # Resize the image to match the input the model will accept
#             img = skimage.transform.resize(img, (28, 28))

#             # Flatten the pixels from 28x28 to 784x0
#             img = img.flatten()

#             # Get prediction of image from classifier
#             predictions = image_classifier.predict([img])

#             # Get the value of the prediction
#             prediction = predictions[0]

#             return flask.render_template('classify_image_twiiter.html', prediction=str(prediction))

#     return(flask.render_template('classify_image_twitter.html'))


if __name__ == '__main__':
    app.run(debug=True)