from flask import Flask, render_template, request, redirect, url_for
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso
import pickle
from my_functions import data_process, LassoFeatureSelector
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('bias_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    
@app.route('/')
def temp():
    return render_template('template.html')

@app.route('/',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        opinions = request.form.get('input1')
        about = request.form.get('input2')
        leaning = request.form.get('input3')
        vid_about = request.form.get('input4')
        
        return redirect(url_for('run_pred',text=opinions, topics = about, biases = leaning, vid_topics = vid_about))

@app.route('/run_pred/<string:text>/<string:topics>/<string:biases>/<string:vid_topics>')
def run_pred(text, topics, biases, vid_topics):
    import numpy as np 
    from sklearn.feature_extraction.text import TfidfVectorizer
    import demoji
    from gensim.parsing.preprocessing import remove_stopwords
    import re
    from textblob import Word
    import nltk
    nltk.download('omw-1.4')
    import subprocess
    from nltk.tokenize import word_tokenize
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import string
    from my_functions import data_process #this is a function that I created including all the steps to clean and vectorize text
    
    
    command = "python -m textblob.download_corpora"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print("Corpora downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    
    opinions = text.split(';')
    about = topics.split(';')
    bias_of_video = biases.split(';')
    topics_of_video = vid_topics.split(';')
    
    processed_opinions = data_process(opinions)
    
    #As soon as i included the text cleaning and word cleaning processes, the deployment started taking too long and crashed
    #Therefore, to have an inaccurate but functional deployment, I tried filling the missing values with zeroes instead of 
    #vectorized text.
    #processed_opinions = []
    #for i in range(len(opinions)):
        #processed_opinions.append([0] * 300)

    category_1 = []
    for i in range(len(about)):
        if about[i] == 0:
            category_1.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #abortion was the dropped column during one hot encoding
        elif about[i] == 1:
            category_1.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        elif about[i] == 2:
            category_1.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            category_1.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    category_2 = []
    for i in range(len(bias_of_video)):
        if int(bias_of_video[i]) == 0:
            category_2.append([0, 0, 0])
        elif int(bias_of_video[i]) == 1:
            category_2.append([1, 0, 0])
        elif int(bias_of_video[i]) == 2:
            category_2.append([0, 1, 0])
        elif int(bias_of_video[i]) == 3:
            category_2.append([0, 0, 1])

    category_3 = []
    for i in range(len(topics_of_video)):
        if int(topics_of_video[i]) == 0:
            category_3.append([0, 0, 0])
        elif int(topics_of_video[i]) == 1:
            category_3.append([1, 0, 0])
        elif int(topics_of_video[i]) == 2:
            category_3.append([0, 1, 0])
        elif int(topics_of_video[i]) == 3:
            category_3.append([0, 0, 1])

    for i in range(len(processed_opinions)):
        processed_opinions[i] = np.hstack((processed_opinions[i], np.array(category_1[i]), np.array(category_2[i]), np.array(category_3[i])))
    
    
    ref_def = pd.read_csv('../ds-final_project/columns_for_model.csv')
    cols = ref_def.columns.tolist()
    processed_opinions = pd.DataFrame(processed_opinions)
    wrong_col = processed_opinions.columns.tolist()
    for i in range(len(wrong_col)):
        processed_opinions.rename(columns={wrong_col[i]:cols[i]} ,inplace=True)
    
        
    model = loaded_model
    pred = loaded_model.predict(np.array(processed_opinions))
    pred_text=[]
    for i in pred:
        if int(i) == 0:
            pred_text.append('liberal')
        if int(i) == 1:
            pred_text.append('centrist')
        else:
            pred_text.append('conservative')
            
    result_strings = []

    for i, value in enumerate(pred_text):
        result_string = f'Opinion {i} is {value}'
        result_strings.append(result_string)
   
    formatted_result = '\n'.join(result_strings)
    
    return formatted_result
    
if __name__ == '__main__':
    app.run()