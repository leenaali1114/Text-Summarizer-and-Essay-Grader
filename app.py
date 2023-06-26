from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request

from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
from essay_grading import grade
import time
import spacy
from flask import Flask,render_template,request,url_for
import pickle
import joblib
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen
#from urllib import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# Reading Time
def readingTime(mytext):
    total_words = len([ token.text for token in nlp(mytext)])
    estimatedTime = total_words/200.0
    return estimatedTime

# Fetch Text From Url
def get_text(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze',methods=['GET','POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
    start = time.time()
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        rawtext = get_text(raw_url)
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
    return render_template('compare_summary.html')

#import Pegasus

@app.route('/comparer',methods=['GET','POST'])
def comparer():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary_spacy)
        # Gensim Summarizer
        final_summary_gensim = summarize(rawtext)
        summary_reading_time_gensim = readingTime(final_summary_gensim)
        # NLTK
        final_summary_nltk = nltk_summarizer(rawtext)
        summary_reading_time_nltk = readingTime(final_summary_nltk)
#         batch = tokenizer.prepare_seq2seq_batch(rawtext, truncation=True, padding='longest').to(torch_device)
#         translated = model.generate(**batch)
#         final_summary_pegasus = tokenizer.batch_decode(translated, skip_special_tokens=True)
#         summary_reading_time_pegasus = readingTime(final_summary_pegasus)
        # Sumy
        final_summary_sumy = sumy_summary(rawtext)
        summary_reading_time_sumy = readingTime(final_summary_sumy) 

        end = time.time()
        final_time = end-start
        return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)


"""@app.route('/grading')
def grading():
    return render_template('grading.html')

@app.route('/grader',methods=['GET','POST'])
def grader():
    if request.method == 'POST':
        essay = request.form['essay']
        predicted_score = grade(essay)
        return render_template('grading.html',ctext=essay,predicted_score=predicted_score)"""

model_list = []
with open("new_1.pickle", "rb") as f:
    for _ in range(pickle.load(f)):
        model_list.append(pickle.load(f))

ohe = model_list[0]
regressor = joblib.load('regressor.sav')

def pred_method(essay,essay_id):
    list1 = []
    list1.append(essay_id)
    sent_count = nltk.sent_tokenize(essay)
    sent_count = len(sent_count)
    list1.append(sent_count)
    #Average sentence length
    word_count = len(essay.split())
    list1.append(int(word_count/sent_count))
    
    #Cleaning the essay
    new_essay = re.sub(r'@[A-Z]{2,}\d?,?\'?s? ?','',essay)   #removing the @location words
    new_essay = re.sub(r'[^a-zA-Z ]','',new_essay)                    #removing the punctuation marks
    new_essay = nltk.word_tokenize(new_essay)
    new_essay = " ".join(new_essay)
    #char count
    char_count = len(re.sub(r'[\s]','',new_essay))
    list1.append(char_count)
    #word_count
    word_count = len(new_essay.split())
    list1.append(word_count)
    #from nltk.stem.wordnet import WordNetLemmatizer 
    #uni-_word_count
    ps = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    uniq_word = len(list(set([ps.stem(word) for word in new_essay.lower().split() if not word in stop_word])))
    list1.append(uniq_word)

    x = new_essay.lower()
    x = nltk.word_tokenize(x)
    pos_tag = nltk.pos_tag(x)
    noun_count = 0
    verb_count = 0
    adjective_count = 0
    adverb_count = 0
    for (word, pos) in pos_tag:
        if pos.startswith('N'):
            noun_count +=1
        elif pos.startswith('V'):
            verb_count +=1
        elif pos.startswith('J'):
            adjective_count += 1
        elif pos.startswith('R'):
            adverb_count +=1
    list1.extend([noun_count , verb_count , adjective_count, adverb_count])
    
    list1 = np.array(list1).reshape(1,-1)
    
    
    
    features_test = ohe.transform(list1).toarray()

    features_test = features_test[:,1:]

    prediction = regressor.predict(features_test)
    return prediction[0]

@app.route('/home',methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/essay01',methods=['GET','POST'])
def essay01():      
    return render_template('essay01.html')

@app.route('/essay02',methods=['GET','POST'])
def essay02():      
    return render_template('essay02.html')

@app.route('/essay03',methods=['GET','POST'])
def essay03():
    return render_template('essay03.html')

@app.route('/essay04',methods=['GET','POST'])
def essay04():      
    return render_template('essay04.html')

@app.route('/essay05',methods=['GET','POST'])
def essay05():      
    return render_template('essay05.html')

@app.route('/essay06',methods=['GET','POST'])
def essay06():      
    return render_template('essay06.html')

@app.route('/essay07',methods=['GET','POST'])
def essay07():      
    return render_template('essay07.html')

@app.route('/essay08',methods=['GET','POST'])
def essay08():      
    return render_template('essay08.html')

@app.route('/result/<int:var>',methods = ['GET','POST'])
def result(var):
    essay_id = int(var)
    essay = str(request.form['essay'])
    score = pred_method(essay,essay_id)
    if essay_id == 1:
        max_score = 12
    elif essay_id == 2:
        max_score = 6
    elif essay_id == 3:
        max_score = 3
    elif essay_id == 4:
        max_score = 3
    elif essay_id == 5:
        max_score = 4
    elif essay_id == 6:
        max_score = 4
    elif essay_id == 7:
        max_score = 30
    elif essay_id == 8:
        max_score = 60
    if score>max_score:
        score = max_score    
    return render_template('result.html',score=score,max_score=max_score)
@app.route('/contact_us')
def contact_us():
    return render_template('contact.html')

@app.route('/how')
def how():
    return render_template("how.html")

@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
	app.run(debug=True)
# Below from jupyternb
#app.run(debug=True, use_reloader=False)
