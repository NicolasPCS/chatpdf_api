from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import tempfile
import shutil
import json
import uvicorn
import os
import csv

load_dotenv()

app = FastAPI()

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

@app.post("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
    global vector_store, chat_history
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        document_path = temp_file.name

    # Process the PDF
    loader = PyPDFLoader(document_path)
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=25)
    docs = text_splitter.split_documents(documents)
    embeddings = AzureOpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./persist")
    vector_store.persist()

    # Add the file upload message to chat history
    #chat_history.append({"role": "system", "content": f"El archivo '{file.filename}' ha sido cargado y procesado correctamente."})

    return {"message": f"El archivo '{file.filename}' ha sido cargado y procesado correctamente."}

@app.post("/send")
async def send_message(request: Request):
    global vector_store, chat_history

    data = await request.json()  # Obtén el JSON del cuerpo de la solicitud
    question = data.get('message')  # Accede al campo 'message'

    print(question)

    if vector_store is None:
        return JSONResponse(content={"message": "No PDF file uploaded"}, status_code=400)

    try:
        embeddings = AzureOpenAIEmbeddings()
        vectordb = Chroma(persist_directory="./persist", embedding_function=embeddings)

        # Verificar el contenido del vector store
        """ print("Contenido del Vector Store:")
        all_docs = vectordb.similarity_search("", k=5)
        for doc in all_docs:
            print(doc.page_content) """

        # Definir el template y las variables de entrada
        template = """
        As a highly knowledgeable assistant specialized in aiding analyst programmers from Bantotal in comprehending client requirements, your task is to respond to questions in Spanish related to the documents delivered by the business analysis area. 
        These documents contain requirement information and a technical solution, including the logic that the program will execute, the database tables involved, and programs related to the solution's development.

        When answering questions, ensure that all information is derived from the PDF document or based on previous chat history. If a question cannot be answered using the information provided in the PDF, honestly state that you cannot answer it. 
        Your responses should be detailed and focused on the question, providing all relevant details necessary for a complete answer without including details beyond the scope of the question.

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
            retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        # Generar el contexto solo con mensajes de usuario y respuestas del asistente
        context_list = []
        documents = vectordb.similarity_search(query=question, k=5)
        for doc in documents:
            context_list.append(doc.page_content)
        context = " ".join(context_list)

        # Pasar la cadena de texto como contexto en el diccionario query
        query = {"context": context, "question": question, "chat_history": chat_history}

        # Intentar obtener la respuesta de la cadena
        result = chain.invoke(query)
        #print("Context: ", query['context'])
        print(query)
        answer = result["answer"]
        
    except Exception as e:
        # Manejar cualquier excepción que ocurra durante la recuperación
        print(f"Error retrieving answer: {e}")
        answer = "Ocurrió un error al procesar la solicitud, por favor, intentalo más tarde."

    # Agregar el nuevo mensaje del usuario al historial del chat
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    print("Chat history: ", chat_history)

    print("Answer final")
    print(answer)

    # Devolver la respuesta JSON
    return JSONResponse(content={"answer": answer})

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