import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import re
import os
import json
from src.utils import read_config
from prompts import gemini_prompts
from src.data_processing import process_data_as_df
import google.generativeai as genai
import google.ai.generativelanguage as glm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from typing import Any, Dict
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
chroma_collection_name = configs['CHROMA_COLLECTION_NAME']


@app.get('/embeddings')
async def creat_embeddings():
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function= len
    )
    
    df_data = process_data_as_df()
    
    reason_loader = DataFrameLoader(df_data, page_content_column='reason')
    reason_data = reason_loader.load()

    # target_loader = DataFrameLoader(organized_result, page_content_column='target')
    # target_data = target_loader.load()

    reason_documents = text_splitter.transform_documents(reason_data)
    print("已為所有文件進行 chunk ",len(reason_documents),"筆")
    
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)
        print(f"Created folder: {vector_db_path}")

    id_list = df_data['source'].to_list()     
    vector_store = Chroma.from_documents(documents=reason_documents, 
                                     persist_directory=vector_db_path, 
                                     collection_name=chroma_collection_name, 
                                     ids=id_list,
                                     embedding=embeddings,
                                     collection_metadata={"hnsw:space": "cosine"})
    

@app.post('/chat')
async def create_chat(request: ChatRequest):
    
    query = "對象包括" + request.query 
    print(query)
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

    vectorstore = Chroma(persist_directory=vector_db_path, collection_name=chroma_collection_name, embedding_function=embeddings)

    metadata_field_info = [

        AttributeInfo(
            name='target',
            description='對象、受文對象、受文者',
            type='string',
        ),
        AttributeInfo(
            name='reason',
            description='被糾正原因、案由',
            type='string',
        ),
        AttributeInfo(
            name='fact',
            description='被糾正的證據、事實、事實與理由',
            type='string',
        ),
        AttributeInfo(
            name='keywords',
            description='被糾正原因中的關鍵字',
            type='string'
        ),
        AttributeInfo(
            name='relationship_between_entities',
            description='被糾正原因中呈現三元組關係的entities(實體)',
            type='string'
        )
    ]

    class CustomSelfQueryRetriever(SelfQueryRetriever):
            def _get_docs_with_query(
                self, query: str, search_kwargs: Dict[str, Any]
            ):
                """Get docs, adding score information."""
                docs, scores = zip(
                    *vectorstore.similarity_search_with_score(query, **search_kwargs)
                )
                for doc, score in zip(docs, scores):
                    if score < 1:
                        doc.metadata["score"] = 1-score
                    elif score >1 :
                        doc.metadata["score"] = 1

                return docs
            
    document_content_description = 'agent'
    retriever = CustomSelfQueryRetriever.from_llm(
            llm,
            vectorstore,
            document_content_description,
            metadata_field_info,
            verbose=True
    )


    try:

        docs = retriever.get_relevant_documents(query)
        response_data = {}
        for i, page in enumerate(docs):
            summary_prompt = PromptTemplate.from_template(gemini_prompts.SUMMARY_PROMPT)
            chain = summary_prompt | llm
            summary_data = {
                "reference": page.page_content
            }

            summary_response = chain.invoke(summary_data)
            
            pattern = r'\*\*摘要：\*\*'
            if re.search(pattern, summary_response.content):
                re.sub(pattern, '', summary_response.content)
                
                
            # conclusion_prompt = PromptTemplate.from_examples(gemini_prompts.CONCLUSION_PROMPT)
            # chain = conclusion_prompt | llm
            # conclusion_data = {
            #     "query" : query,
            #     "reference": page.page_content
            # }
            # conclusion_response = chain.invoke(conclusion_data)

            response_data[i] = {
                "summary": summary_response.content,
                "source": page.metadata['source'],
                "score": f"{page.metadata['score']:.2f}",
                "target": page.metadata['target'],
                # "conclusion": conclusion_response.content
            }


    except:
        return print('no answer found.')
        
    return response_data
    # except:
    #     return print('no answer found.')

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