from flask import Flask, request
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SummaryIndex
from llama_index.readers.mongodb import SimpleMongoReader
import os
from flask_cors import CORS
import logging
import sys

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000"
])

@app.route("/")
def home():
    return "Hello World!"

def initialize_index():
    global index
    host = "ip"
    port = 27017
    db_name = "db"
    collection_name = "collection"
    # query_dict is passed into db.collection.find()
    query_dict = {}
    field_names = ["content"]
    reader = SimpleMongoReader(host, port)
    documents = reader.load_data(
        db_name, collection_name, field_names, query_dict=query_dict
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    index = SummaryIndex.from_documents(documents)


@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    print(str(response))
    return str(response), 200


if __name__ == "__main__":
    initialize_index()
    app.run(host="0.0.0.0", port=5601)