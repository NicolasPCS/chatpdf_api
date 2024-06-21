from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from PyPDF2 import PdfFileReader

from dotenv import load_dotenv
from pydantic import BaseModel
import tempfile
import shutil
import uvicorn
import os
import csv
import faiss

from chain import chain

load_dotenv()

app = FastAPI()

# Directorio persistente local en App Service o por defecto
HOME_DIR = os.getenv("HOME", ".")  # Utilizar el directorio actual si HOME no está definido
PERSIST_DIR = os.path.join(HOME_DIR, "persist")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Static HTML file handling using Jinja2
templates = Jinja2Templates(directory="templates")

# Global variables
vector_store = None
chat_history = []
chat_histories = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("pagina_web.html", {"request": request})

@app.get("/health")
def health_check():
    return 'OK'

class ChatRequest(BaseModel):
    user_id: str
    message: str

# Con FAISS
@app.post("/uploadFile")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    global vector_store

    user_dir = os.path.join(PERSIST_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    print(user_dir)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            document_path = temp_file.name

        # Process the PDF
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = AzureOpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Guardar el vector store en el directorio del usuario
        index_file_path = os.path.join(user_dir, "vector_store.index")
        faiss.write_index(vector_store.index, index_file_path)

        #print(vector_store.index.ntotal)
        # Print the content of the documents in the vector store
        print("Documentos cargados en el vector store:")
        """ for doc in docs:
            print(doc)  # Imprime los primeros 200 caracteres de cada documento """
        
        question = "Describe detalladamente todas las secciones del documento"
        result = agent(question)
        #print(query)
        answer = result["answer"]

        query = {
            "message": f"El archivo '{file.filename}' ha sido cargado y procesado correctamente.",
            "description": answer
            }

        return JSONResponse(content=query)
    
    except Exception as e:
        # Capturar y mostrar el error detallado
        error_message = f"Error procesando el archivo: {str(e)}"
        print(error_message)
        return JSONResponse(content={"message": error_message}, status_code=500)

# Con FAISS
@app.post("/send")
async def send_message(request: ChatRequest):
    global chat_history

    user_id = request.user_id
    question = request.message
    user_dir = os.path.join(PERSIST_DIR, user_id)
    index_file_path = os.path.join(user_dir, "vector_store.index")

    #print(request)

    if not os.path.exists(index_file_path):
        return JSONResponse(content={"message": f"No se ha cargado ningún archivo PDF para este usuario. Usuario Nro. {user_id}. Usuario Dir. {user_dir}"}, status_code=400)

    try:
        # Intentar obtener la respuesta de la cadena
        result = agent(question)
        #print(query)
        answer = result["answer"]
        
    except Exception as e:
        # Manejar cualquier excepción que ocurra durante la recuperación
        print(f"Error retrieving answer: {e}")
        answer = "Ocurrió un error al procesar la solicitud, por favor, intentalo más tarde."

    # Agregar el nuevo mensaje del usuario al historial del chat
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    #print("Chat history: ", chat_history)
    #print("Answer final")
    #print(answer)

    # Devolver la respuesta JSON
    return JSONResponse(content={"answer": answer})

def agent(question):
    # Definir el template y las variables de entrada
    template = """
        Task: You are a helpful assistant, an expert in the core of Bantotal.
        Your task is to assist Bantotal's analyst programmers in understanding the development of a requirement in Genexus. 
        You will provide relevant information based on the user's requests. The user will upload a document containing all the information about the requirement and will ask you questions about this file.

        Instructions:

        - You must answer the user's questions in SPANISH clearly, precisely, and in detail, ensuring that you use only the information provided in the document, the chat history, and any prior knowledge about Genexus and Bantotal that you possess.
        - If a question cannot be answered based on the information in the document or your knowledge, you should ask the user to provide more information or context. Otherwise, honestly state that you cannot answer the question.
        - Be detailed in your responses, but stay focused on the question. Add all useful details to provide a complete answer, but do not include details that are outside the scope of the question.
        - Below is an example of the structure that a requirements document may have:
        Example of the structure:
            - Objective: Helps analyst programmers understand the objective of the requirement.
            - Scope: Helps analyst programmers understand the scope of the requirement.
            - Requirement Description: Provides a detailed understanding of the requirement.
            - Proposed Solution:
                - Functional Solution/Functional Solution Description: Considerations before starting the development of the requirement.
                - Technical Solution/Technical Solution Description: Includes the solution in pseudocode or critical technical aspects for development in Genexus.
            - Validation Matrix: Validates that the program meets the functionalities described in the objective of the requirement.
        In most cases, this structure is followed. However, some documents may not strictly adhere to this structure. Make sure to answer the user's questions using only the information contained in the PDF, which is stored in a vector database.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """

    input_variables = ["context", "question"]
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=input_variables)

    # Configuración del modelo LLM y embeddings
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k", 
        temperature=0
    )

    # Crear la cadena de conversación con recuperación utilizando un prompt personalizado
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Generar el contexto solo con mensajes de usuario y respuestas del asistente
    context_list = []
    documents = vector_store.similarity_search(query=question, k=5)
    for doc in documents:
        #print(doc)
        context_list.append(doc.page_content)
    context = " ".join(context_list)

    # Pasar la cadena de texto como contexto en el diccionario query
    query = {"context": context, "question": question, "chat_history": chat_history}

    return chain.invoke(query)

#----------------------------------------------------------------------------------------

@app.post("/new_chat")
async def handle_new_chat(request: Request):
    global chat_histories
    data = await request.json()
    tab_id = data['tabId']
    chat_histories[tab_id] = []
    return JSONResponse(content={'status': f'new chat for {tab_id}'})

@app.post("/feedback")
async def feedback(request: Request):
    global chat_history
    data = await request.json()
    feedback = data['feedback']
    tab_id = data['tabId']
    positive = int(data["positive"]) * 2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta", "Respuesta", "Seccion", "Positivo", "Comentario"])
        writer.writerow([chat_history[-2]['content'], chat_history[-1]['content'], "selected_option", positive, feedback])
    print([chat_history[-2]['content'], chat_history[-1]['content'], feedback])
    return JSONResponse(content={'status': 'Feedback recibido'})

if __name__ == '__main__':
    load_dotenv()
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', "127.0.0.1")
    uvicorn.run(app, host=host, port=port)