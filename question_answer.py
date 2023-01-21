import numpy as np
import openai
import pandas as pd
import pickle
from flask import Flask, jsonify
from flask import request

from flask import Flask, jsonify, request
from flask_cors import CORS



import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
#import tiktoken
openai.api_key = "<<Your OPEN AI keys>>"
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
app = Flask(__name__)

CORS(app)
df1 = pd.read_csv('c:/shashank/test.csv')

df1 = df1.set_index(["title", "heading"])
print(f"{len(df1)} rows in the data.")
#print(pd.__version__)
#print(df1.iloc[1])

datafile_path = "c:/shashank/k8s_embedding.csv"

df = pd.read_csv(datafile_path)

df["content"] = df.content.apply(eval).apply(np.array)


def generate_embedding():
    df['content'] = df.content.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

    df.to_csv('/content/mydrive/MyDrive/k8s_embedding.csv')

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
    
    prompt = request.json.get("input_text")
    print(prompt)
    prompt1 = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don’t know"

    Context:

    """
    prompt2= "Q:"+prompt+" A:"

    print (prompt2)
    prompt=prompt1+search_embedding(df, prompt, n=3)+prompt2
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
    response=jsonify(text=answer)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run()


