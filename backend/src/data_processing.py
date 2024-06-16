import os
import re
import pandas as pd
import json
from src.utils import read_config
from langchain_community.document_loaders import PyPDFLoader
from src.FastTextRank4Word import FastTextRank4Word
from src.FastTextRank4Sentence import FastTextRank4Sentence
from langchain.prompts import PromptTemplate
from prompts import gemini_prompts
import google.generativeai as genai
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

configs = read_config('.env/configs.json')
api_key =configs['g_key']
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)


def read_pdf(files_list, data_path):
    content_dict = {}

    for file_name in files_list:
        file_path = data_path + file_name
        loader = PyPDFLoader(file_path)
        content = loader.load()
        whole_pdf = ""
        for i in range(len(content)):
            whole_pdf += content[i].page_content

        whole_pdf = re.sub(r'[：。\n\s]', '', whole_pdf)
        content_dict[file_name] = whole_pdf

    return content_dict

def content_extraction(input, pattern):

    # Search using the pattern
    match = re.search(pattern, input, re.S)

    if match:
        result = match.group(1).strip()  # Use strip() to remove any leading/trailing whitespace
        return result
    else:
        print("No match found")
        pass

def pattern_extraction(input):

    pattern = r"壹、被糾正機關(.*)貳、案由"
    target = content_extraction(input, pattern)
    pattern = r"貳、案由(.*?)參、事實.*?理由"
    reason = content_extraction(input, pattern)
    pattern = r"參、事實.*?理由(.*)"
    fact = content_extraction(input, pattern)


    return {"target": target, "reason": reason, "fact": fact}

def word_sentence_extraction(input, kw_num=10, ks_num=1):

    kw_model = FastTextRank4Word(tol=0.0001, window=5)
    kw=[kw_model.summarize(input['reason'], kw_num)][0]

    ks_model = FastTextRank4Sentence(use_w2v=False,tol=0.0001)
    ks=ks_model.summarize(input['fact'], ks_num)

    return { 
        "target": f"{input['target']}",
        "reason": f"{input['reason']}",
        "fact" : f"{input['fact']}",
        "keywords": f"{str(kw)}"
        }
    
def content_organize(input):

    result = {}

    for i, (name, content) in enumerate(input.items()):
        try:
            split = pattern_extraction(content)
            temp = keyword_extraction(split)
        except:
            print(f"{name} is problematic.")
            pass
        result[name] = temp

    print(f"{i+1} contents have been organized.")
    return result

def creat_knowledge_graph(df: pd.DataFrame):
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    
    kg_result = df.copy()
    for i, row in kg_result.iterrows():
        
        triple_prompt = PromptTemplate.from_template(gemini_prompts.TRIPLE_PROMPT)
        chain = triple_prompt | llm
        triple_data = {
            "reference": row['reason']
        }

        triple_response = chain.invoke(triple_data)

        triple_response_result = re.sub(r'[```json\n]', '', triple_response.content).replace("實體", "entity").replace("關係", "relationship").split("、")[0]

        to_json_prompt = PromptTemplate.from_template(gemini_prompts.TO_JSON_PROMPT)
        chain = to_json_prompt | llm
        data = {
            "reference": triple_response_result
        }

        to_json_response = chain.invoke(data)
        
        
        triple_json_result = re.sub(r'[\n]','',to_json_response.content)

        # triple_json_result = json.loads(f'[{triple_json_result}]')
        
        kg_result.loc[kg_result.index==i, 'relationship_between_entities'] = str(triple_json_result)
        
    return kg_result
        

def process_data_as_df() -> pd.DataFrame:
    
    configs = read_config('.env/configs.json')
    DATA_PATH = configs["DATA_PATH"]
    # DB_PATH = configs['DB_PATH']
    files_list = [file for file in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, file))]
    
    pdf_contents = read_pdf(files_list, DATA_PATH)
        
    organized_result = content_organize(dict(list(pdf_contents.items())))
    
    df = pd.DataFrame(organized_result).transpose()
    df = df.reset_index().rename(columns={'index':'source'})
    
    df_with_kg = creat_knowledge_graph(df)
    
    return df_with_kg