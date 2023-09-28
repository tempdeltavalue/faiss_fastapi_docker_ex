from typing import Union
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
import faiss

app = FastAPI()

def search(index, data, encoder, query, k=1):

    query_vector = encoder.encode([query])
    top_k = index.search(query_vector, k)
    print(top_k)
    return [
        data[_id] for _id in top_k[1][0]
    ]


@app.get("/")
def read_root():
    path = 'app/faiss.index'

    index = faiss.read_index(path)

    encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    data = [
        'Love is beautiful',
        'Pain is bad',
    ]


    txt = search(index, data, encoder, "What is beautiful?")


    return {"Hello": txt[0]}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}