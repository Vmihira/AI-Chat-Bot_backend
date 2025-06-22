import chromadb
from text_chunker import TextChunker
from dotenv import load_dotenv
import os   
import semchunk
import tiktoken

load_dotenv()

client = chromadb.PersistentClient()


def insert_document(collection_name, text):


    chunker = semchunk.chunkerify('gpt-4', chunk_size=50) # chunk_size in tokens

    chunks = chunker(text)

    collection = client.get_or_create_collection(name=collection_name)

    ids = [str(hash(chunk)) for chunk in chunks]
    

    collection.add(documents=chunks, ids= ids)


def query_document(collection_name, query_text, n_results=5):
    collection = client.get_or_create_collection(name=collection_name)

    response = collection.query(
    query_texts=[query_text],
    n_results=10,
    )

    print("==========="*10)
    print(f"Query response: {response}")

    documents = response["documents"]

    return ('\n').join(documents[0]) 

def get_collections():
    return client.list_collections()



# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate(user_query):

    print(user_query)

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text= user_query),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are chat bot who answer user questions using youre knowledge and knowledge that is provided to you a reference"""),
        ],
    )

    result = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
    
        result += chunk.text
    
    return result

