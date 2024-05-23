# IMPORTS
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
import os

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import PromptValue

from chain import chain

import uvicorn

app = FastAPI()

# Static HTML file handling using Jinja2
templates = Jinja2Templates(directory="templates")

# Global variables
ultima_respuesta = {}
ultima_pregunta = {}
chat_histories = {}
model = 0

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("pagina_web.html", {"request": request})

@app.get("/health")
def health_check():
    return 'OK'

# JS Associated functions
async def generate_data(message, tab_id):
    global ultima_respuesta
    global ultima_pregunta
    global model
    chat_history = chat_histories[tab_id]
    chat_history_string = ""

    for m in chat_history:
        chat_history_string += f"{m.type}: {m.content} \n"
    resp = ""

    if model == 0:
        async for chunk in chain.astream({"input": message, "chat_history": chat_history}):
            if chunk.content:
                content = chunk.content.replace("\n", "||")
                resp += chunk.content
                yield f"data: {content}\n\n"
    if model == 1:
        return
    
    ultima_respuesta[tab_id] = resp
    chat_history.append(HumanMessage(content = message))
    chat_history.append(AIMessage(content = resp))

    if len(chat_history) >= 6:
        chat_history = chat_history[-6:]
    chat_histories[tab_id] = chat_history
    yield "data: done\n\n"

@app.post("/send")
async def send(request: Request):
    global ultima_pregunta

    data = await request.json()
    tab_id = data['tabId']
    if tab_id not in chat_histories:
        chat_histories[tab_id] = []
    ultima_pregunta[tab_id] = data['message']
    return JSONResponse(content = {'status': 'success'})

@app.get("/stream")
def stream(request: Request):
    tab_id = request.query_params['tabId']
    print(f"Respondiendo pregunta de {tab_id}")
    global ultima_pregunta
    return StreamingResponse(generate_data(ultima_pregunta[tab_id], tab_id), media_type='text/event-stream')

@app.get("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file with automatic deletion upon closing
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            document_path = temp_file.name

            # Perform processing or analysis using document_path
            # ... (your logic here)

            return JSONResponse(content={"status": "success", "message": "File uploaded successfully"})

    except Exception as e:
        print(f"Error uploading file: {e}")
        return JSONResponse(content={"status": "error", "message": "Error uploading file"})

@app.post("/new_chat")
async def handle_new_chat(request: Request):
    global chat_histories
    data = await request.json()
    tab_id = data['tabId']
    chat_histories[tab_id] = []
    return JSONResponse(content={'status': f'new chat for {tab_id}'})

@app.post("/feedback")
async def feedback(request: Request):
    global ultima_pregunta
    global ultima_respuesta

    data = await request.json()
    feedback = data['feedback']
    tab_id = data['tabId']
    positive = int(data["postitive"]) * 2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta", "Respuesta", "Seccion", "Positivo", "Comentario"])
        writer.writerow([ultima_pregunta[tab_id], ultima_respuesta[tab_id], "selected_option", positive, feedback])
    print([ultima_pregunta[tab_id], ultima_respuesta[tab_id], feedback])
    return JSONResponse(content={'status': 'Feedback recibido'})

if __name__ == '__main__':
    load_dotenv()
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', "127.0.0.1")
    uvicorn.run(app, host=host, port=port)