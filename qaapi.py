import numpy as np
import openai
import pandas as pd
import pickle
from flask import Flask, jsonify
from flask import request

from flask import Flask, jsonify, request
from flask_cors import CORS


import json
from promptify.models.nlp.openai_model import OpenAI
from promptify.prompts.nlp.prompter import Prompter
from pprint import pprint
from IPython.display import Markdown, display
from IPython.core.display import display, HTML

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
#import tiktoken
openai.api_key = ""
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
app = Flask(__name__)

CORS(app)
df1 = pd.read_csv('c:/shashank/api.csv')
history = ""

df1 = df1.set_index(["title", "heading"])
print(f"{len(df1)} rows in the data.")
#print(pd.__version__)
#print(df1.iloc[1])

datafile_path = "c:/shashank/api_embedding.csv"

df = pd.read_csv(datafile_path)

df["content"] = df.content.apply(eval).apply(np.array)









def search_embedding(df, product_description, n=3, pprint=True):

    embedding = get_embedding(

    product_description,

    engine="text-embedding-ada-002"

    )
    df["similarities"] = df.content.apply(lambda x: cosine_similarity(x, embedding))

    max_index = df["similarities"].idxmax()
    print(max_index)

    response=df1.iloc[max_index,df1.columns.get_loc('content')]

    return response


@app.route('/create_embedding', methods=['POST','OPTIONS'])
def create_embedding_route():
    print("entered embedding route")
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    heading = request.json['heading']
    title = request.json['title']
    content = request.json['content']
    embedding_length = request.json['embedding_length']
    processed_text = process_text(heading, title, content, embedding_length)
    response = jsonify({'text': processed_text})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def process_text(heading, title, content, embedding_length):
    #create a dataframe 
    data = {'title': [title], 'heading': [heading], 'content': [content], 'tokens': [embedding_length]}
    df = pd.DataFrame(data)
    print("printing new df")
    print(df)
    
    df_existing = pd.read_csv('c:/shashank/test.csv')
    
    
    
    print(df_existing)
# Append the new dataframe to the existing dataframe
    df_existing = pd.concat([df_existing, df],ignore_index=True,axis=0)
    print("printing modified test.csv ")
    print(df_existing)
# Write the concatenated dataframe back to the CSV file
    df_existing.to_csv('c:/shashank/test.csv')
    
    df_existing1 = pd.read_csv('c:/shashank/k8s_embedding.csv')
    #df.to_csv('c:/shashank/test.csv',mode='a', header=False,index=False)
    df['content'] = df.content.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    
    
    
    
    

# Append the new dataframe to the existing dataframe
    df_existing1 = pd.concat([df_existing1, df], ignore_index=True,axis=0)
    
    df_existing1.to_csv('c:/shashank/api_embedding.csv')
    print (df)
    # your logic to process the text here

@app.route('/generate_questions', methods=['POST','OPTIONS'])
def generate_questions():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    

    content_type = request.headers.get('Content-Type')
    print(content_type)
    if (content_type == 'application/json'):
        #json = request.get_json(force=True)
        print("got json")
    else:
        return 'Content-Type not supported!'
    
    input = request.json.get("input_text")
    content=search_embedding(df, input, n=3)
    model = OpenAI(openai.api_key)
    nlp_prompter = Prompter(model)
    context=content


    result = nlp_prompter.fit('qa_gen.jinja',
                                      text_input=context,
                                      domain="software",
                                      total_questions=5,
                                      max_QA_tokens=10
                                     )


    #pprint(eval(result['text']))
    response=jsonify(text=result['text'])
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



@app.route('/generate_explanation', methods=['POST','OPTIONS'])
def generate_explanation():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    

    content_type = request.headers.get('Content-Type')
    print(content_type)
    if (content_type == 'application/json'):
        #json = request.get_json(force=True)
        print("got json")
    else:
        return 'Content-Type not supported!'
    
    input = request.json.get("input_text")
    content=search_embedding(df, input, n=3)
    model = OpenAI(openai.api_key)
    nlp_prompter = Prompter(model)
    context=content


    result = nlp_prompter.fit('explain.jinja',
                                      text_input=context,
                                      domain="software",
                                      token_length = None
                                     )





    #pprint(eval(result['text']))
    response=jsonify(text=result['text'])
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



@app.route('/generate_text', methods=['POST','OPTIONS'])
def generate_text_route():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    

    content_type = request.headers.get('Content-Type')
    print(content_type)
    if (content_type == 'application/json'):
        #json = request.get_json(force=True)
        print("got json")
    else:
        return 'Content-Type not supported!'
    
    global history
    
    prompt = request.json.get("input_text")
    input = request.json.get("input_text")
    #history += input
    print(history)
    prompt1 = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don’t know"

    Context:

    """
    prompt2= "Q:"+prompt+" A:"

    print (prompt2)
    prompt=prompt1+search_embedding(df, prompt, n=3)+history+prompt2
    answer=openai.Completion.create(

        prompt=prompt,

        temperature=0,

        max_tokens=300,

        top_p=1,

        frequency_penalty=0,

        presence_penalty=0,

        model=COMPLETIONS_MODEL

        )["choices"][0]["text"].strip(" \n")
    print(answer)
    history += input
    history += answer
    response=jsonify(text=answer)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run()


