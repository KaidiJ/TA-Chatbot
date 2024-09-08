import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from threading import Thread
import json
from bs4 import BeautifulSoup
from threading import Thread
import sqlite3
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import requests
import time


app = Flask(__name__)
CORS(app)
conversation_chain = None

db_texts = None
db_vectors = None


def initialize_data():
    global db_texts, db_vectors, db_links
    db_texts, db_vectors, db_links = read_vectors_from_db()
    print("loading database")

def get_db_connection(db_path="text_vectors.db"):
    """ Connect to the SQLite database. """
    conn = sqlite3.connect(db_path)
    return conn

def read_vectors_from_db():
    """Read text chunks, corresponding vectors and video links (if available) from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text, vector, video_time_link FROM text_vectors")
    db_data = cursor.fetchall()
    conn.close()
    texts, vectors, links = zip(*db_data)
    vectors = [np.fromstring(vector[1:-1], sep=', ') for vector in vectors]  # Convert string back to numpy array
    return texts, np.array(vectors), links


# Embedding pdf
def get_pdf_text(pdf_path):
    """Extract text from PDF file given path."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # 添加或""以避免None
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def save_text_to_db(text, db_path="piazza_posts.db"):
    """Save the text to the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')
    try:
        c.execute("INSERT INTO posts (content) VALUES (?)", (text,))
        conn.commit()
        print("Text saved successfully to database.")
    except sqlite3.IntegrityError as e:
        print(f"Error saving text to database: {e}")
    finally:
        conn.close()

def run_command_line_interaction():
    global conversation_chain
    while True:
        embed_pdf = input("Do you want to embed a PDF? (y/n): ")
        if embed_pdf.lower() == 'y':
            pdf_path = input("Enter the path to the PDF: ")
            if os.path.exists(pdf_path):
                pdf_text = get_pdf_text(pdf_path)
                save_text_to_db(pdf_text)
                print("PDF text embedding process completed.")
            else:
                print("PDF file does not exist. Please check the path.")
        elif embed_pdf.lower() == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


# Get Embedding:
def get_embedding(text, model="text-embedding-3-small"):
    """
     Get the embedding vector of the given text.

     :param text: The text to get the embedding.
     :param model: The name of the embedding model used.
     :return: A list containing embedding vectors.
    """
    openai_api_key = ""  # API key
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    data = {"input": text, "model": model}
    response = requests.post("https://api.openai.com/v1/embeddings", json=data, headers=headers)

    if response.status_code == 200:
        embedding_vector = response.json()["data"][0]["embedding"]
        return embedding_vector
    else:
        print("Failed to get embedding:", response.text)
        return None


# Compare similarity:
def compare_user_question_to_db_vectors(user_question, db_vectors, db_texts, db_links, top_k=15):
    user_question_vector = get_embedding(user_question)

    if user_question_vector is not None:
        similarities = cosine_similarity([user_question_vector], db_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        top_texts = []
        top_links = []

        for index in top_indices:
            top_texts.append(db_texts[index])
            top_links.append(db_links[index])
            if index > 0:
                top_texts.append(db_texts[index - 1])
                top_links.append(db_links[index - 1])
            if index < len(db_texts) - 1:
                top_texts.append(db_texts[index + 1])
                top_links.append(db_links[index + 1])

        for i in range(min(5, len(top_links))):
            if top_links[i]: # Assume that the way to detect video links is to check whether the link contains specific characters or fields
                # Find the video link and return the text of the link and the text before and after it
                linked_texts = [top_texts[i]]
                linked_texts_indices = [i]

                # add text before it
                if i > 0:
                    linked_texts.insert(0, top_texts[i - 1])
                    linked_texts_indices.insert(0, i - 1)

                # add text after it
                if i < len(top_texts) - 1:
                    linked_texts.append(top_texts[i + 1])
                    linked_texts_indices.append(i + 1)

                print(linked_texts)
                return linked_texts, top_links[i]

        # Returns an expanded text list and None if the video link is not found in the top results
        return top_texts, None
    else:
        print("Failed to vectorize user question.")
        return [], None


# Merge text in get_vectorstore function
def get_vectorstore(texts):
    openai_api_key = ""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)  # 注意这里是单个字符串列表
    return vectorstore


def get_conversation_chain(vectorstore):
    openai_api_key = ""
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain


def start_cli_thread():
    cli_thread = Thread(target=run_command_line_interaction)
    cli_thread.daemon = True
    cli_thread.start()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        start_time_total = time.time()
        user_message = request.json["message"]

        start_time_top_texts = time.time()
        top_texts, video_link = compare_user_question_to_db_vectors(user_message, db_vectors, db_texts, db_links)
        end_time_top_texts = time.time()
        print(f"time to get top_texts: {end_time_top_texts - start_time_top_texts} s")

        if video_link:
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            chatbot_response = response[
                                   "answer"] + f"\n\nHere's a related video you might find helpful: {video_link}\n\n"
        else:
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            chatbot_response = response["answer"]

        end_time_total = time.time()
        print(f"Total time from receiving question to giving response: {end_time_total - start_time_total} s")

        print(f"Chatbot response: {chatbot_response}")

        return jsonify({"response": chatbot_response})
    except KeyError:
        return jsonify({"error": "Invalid request data"}), 400


if __name__ == "__main__":
    load_dotenv()
    start_cli_thread()
    initialize_data()
    app.run(debug=False, port=5001)