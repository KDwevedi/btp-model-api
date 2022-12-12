from fastapi import FastAPI, Form
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import re

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# origins = [
#     "https://127.0.0.1:*",
#     "http://127.0.0.1:*",
#     "http://127.0.0.1",
#     "https://127.0.0.1",
#     "*"

# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     #allow_origin_regex='http://127.0.0.1:*',
#     allow_credentials=False,
#     expose_headers=['Access-Control-Allow-Origin'],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


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