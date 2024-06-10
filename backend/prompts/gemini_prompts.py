SUMMARY_PROMPT="""
### INSTRUCTION: 你是一位資深的監察院案件資料專家。你的目標是對以下 REF 資料進行摘要，若沒摘要好，外婆會很傷心。不要列出"**摘要:**"等字樣。`
### REF: {reference}
### ASSISTANT: """
    
TRIPLE_PROMPT="""
### INSTRUCTION: 三元組指的是：(實體)--關係--(實體)。(實體)通常是特定的人、組織或物體，(關係)則是描述兩個實體之間的關係，例如(立法委員)--制定--(法律)。將 REF 的内容以三元組方式描述並以Json格式呈現，給我三組。
### REF: {reference}
### ASSISTANT: """

TO_JSON_PROMPT="""
### INSTRUCTION: 將以下內容調整成符合JSON規範的格式，以及補上entity1或是entity2，不要顯示JSON字樣，注意必須符合正規的JSON格式，否則外婆會很難過。
### REF: {reference}
### ASSISTANT: """

CONCLUSION_PROMPT="""
### INSTRUCTION: 說明查詢的內容{query}與 REF 中的之間的關係。若找不到合理的關係則說沒有明確的相關性，不要亂說
### REF: {reference}
### ASSISTANT: 
"""