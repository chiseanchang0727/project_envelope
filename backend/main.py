import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils import read_config
import os
from src.data_processing import process_data_as_df
import google.generativeai as genai
import google.ai.generativelanguage as glm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from pydantic import BaseModel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ChatRequest(BaseModel):
    query: str

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


configs = read_config('.env/configs.json')
api_key =configs['g_key']
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)
vector_db_path = configs['VECTORDB_PATH']
# chroma_collection_name = "langchain"


@app.post('/embeddings')
async def creat_embeddings():
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function= len
    )
    
    organized_result = process_data_as_df()
    
    reason_loader = DataFrameLoader(organized_result, page_content_column='reason')
    reason_data = reason_loader.load()

    # target_loader = DataFrameLoader(organized_result, page_content_column='target')
    # target_data = target_loader.load()

    reason_documents = text_splitter.transform_documents(reason_data)
    # print("已為所有文件進行 chunk ",len(reason_documents),"筆")
    
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    

    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)
        print(f"Created folder: {vector_db_path}")
        
    vector_store = FAISS.from_documents(reason_documents, embeddings)
    vector_store.save_local(folder_path=vector_db_path)
    

@app.post('/chat')
async def create_chat(request: ChatRequest):
    
    query = request.query
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    
    vectorstore = FAISS.load_local(folder_path=vector_db_path, allow_dangerous_deserialization=True, embeddings=embeddings)
    
    # query = "行政院原住民委員會的相關資訊"

    embedding_vector = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(embedding_vector, k=3)
    response_data = {}
    for i, page in enumerate(docs):
        # match_content_list.append(page.page_content)
        # reference[i] = {
        #     "reason": page.page_content,
        #     "metadata": page.metadata
        # }

        template = """
        ### INSTRUCTION: 你是一位資深的監察院案件資料專家。你的目標是對以下 REF 資料進行摘要。只要提供摘要結果就好。`
        ### REF: {reference}
        ### ASSISTANT: """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        data = {
            "reference": page.page_content
        }

    
        llm_response = chain.invoke(data)

        response_data[i] = {
            "summary": llm_response.content,
            "source": page.metadata['document']
        }

    return JSONResponse(response_data)

if __name__ == "__main__":
    

    configs = {
    "endpoint": "http://localhost:8000/",
    "host": "0.0.0.0",
    "port": 8000,
    "year_region": 5
    }

    host = configs.get('host', '127.0.0.1')  # Default to 127.0.0.1 if not specified
    port = configs.get('port', 8000)
    
    uvicorn.run(app, host=host, port=port)