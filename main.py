from fastapi import FastAPI, Form
from transformers import pipeline
from starlette.responses import HTMLResponse 
import re


app = FastAPI()
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

def preProcess_data(text): #cleaning the data
    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="500000" name="text" value="Text to be summarized"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    text = preProcess_data(text)
    summary = summarizer(text, max_length=500, min_length=30, do_sample=False) #making summary
    return summary