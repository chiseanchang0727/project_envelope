import os
import re
import pandas as pd
import json
from src.utils import read_config
from langchain_community.document_loaders import DataFrameLoader
from src.FastTextRank4Word import FastTextRank4Word
from src.FastTextRank4Sentence import FastTextRank4Sentence




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
        "metadata" : { "kw" : f"{kw}", "ks" : f"{ks}"}
        }
    
def content_organize(input):

    result = {}

    for i, (name, content) in enumerate(input.items()):
        split = pattern_extraction(content)
        temp = word_sentence_extraction(split)
        result[name] = temp

    print(f"{i+1} contents have been organized.")
    return result

def process_data_as_df() -> pd.DataFrame:
    
    configs = read_config('.env/configs.json')
    DATA_PATH = configs["DATA_PATH"]
    # DB_PATH = configs['DB_PATH']
    files_list = [file for file in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, file))]
    
    pdf_contents = read_pdf(files_list, DATA_PATH)
        
    organized_result = content_organize(dict(list(pdf_contents.items())))
    
    df = pd.DataFrame(organized_result).transpose()
    df = df.reset_index().rename(columns={'index':'document'})
    
    return df