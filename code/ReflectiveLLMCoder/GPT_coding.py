from openai import OpenAI,AzureOpenAI
import httpx
import time
import pandas as pd
import pdb
import glob
import numpy as np

import functions

GPT_MODEL = "gpt-4o"
PROMPT_COST_1M = 5
COMPLETION_COST_1M = 15
MAX_TOKENS = 1200 

SYSTEM = '''
    As a computational social science expert in human mobility and segregation, your task is to perform open coding on a corpus. 
    The corpus contains explanations of how the real-world racial/ethnic composition of visitors to Points of Interests (POIs) deviate from the residential composition (only considers the geographical factor), by deriving insights from Yelp review data into cultural, racial, ethnic, and socioeconomic factors.
'''

PROMPT_CODING_FIRST = '''
    Based on the following corpus, please follow four steps below to construct codings that capture different aspects of POI characteristics that are helpful for predicting the visiting tendencies of different racial/ethnic groups.
    Step 1: Extract all mentioned aspects in the corpus. For example, in the sentence "**Cuisine and Menu Offerings**: Reviews highlight the shop's traditional Latin American dishes, such as tacos or empanadas, which may attract Hispanic visitors who prefer authentic cultural foods.", you should extract "Cuisine and Menu Offerings" as an aspect.
    Step 2: Combine similar aspects by looking at both the aspects and the explanation sentences. For example, if two aspects contain the same word, such as "Cultural Relevance and Appeal" and "Inclusivity and Cultural Representation", they should be combined together as one code.
    Step 3: Remove aspects that are not insightful. For example, "Visitor Analysis" is not an insightful aspect at all, as it doesn't reflect any specific features we can look for to understand visitor preferences.
    Step 4: Refine the explanation sentence to be an insightful and generalizable interpretation of the aspect without explicitly mentioning specific racial/ethnic groups.
    Your coding should be comprehensive (cover as many aspects mentioned in the corpus as possible) and orthogonal (try your best to derive aspects that overlap the least with each other).
    Keep the number of aspects no more than {max_num_aspects}.
    Strictly structure your answer in this format: aspect: content; aspect: content; ... (Each aspect is a phrase describing the captured feature. Each content is a one-sentence explanation of the aspect.) Do not include redundant sentences.
    
    **Corpus**
    {corpus}
'''
#Step 4: Refine the explanation sentence for each aspect, to explicitly mention what condition leads to the attraction/repulsion of which racial/ethnic groups. For example, "Nice dishes appeals to visitors who enjoy food, contributing to a diverse visitor composition" needs to be refined as "a diverse visitor composition" is super ambiguous. Please specify something like "Mentions of tacos or empanadas appeal to Hipanics while mentions of noodles or dumplings appeal to Asian" (just an example, not necessarily true).

PROMPT_CODING = '''
    Your current coding of the previously seen corpus is: {current_coding}
    Given the new corpus, please revise and refine your current coding to be comprehensively helpful for predicting the visiting tendencies of different racial/ethnic groups, following five steps below.
    Step 1: Extract all mentioned aspects in the new corpus. For example, in the sentence "**Cuisine and Menu Offerings**: Reviews highlight the shop's traditional Latin American dishes, such as tacos or empanadas, which may attract Hispanic visitors who prefer authentic cultural foods.", you should extract "Cuisine and Menu Offerings" as an aspect.
    Step 2: Combine similar aspects in the new corpus by looking at both the aspects and the explanation sentences. For example, if two aspects contain the same word such as "Cultural Relevance and Appeal" and "Inclusivity and Cultural Representation", they should be combined together as one code.
    Step 3: Remove aspects that are not insightful. For example, "Visitor Analysis" is not an insightful aspect at all, as it doesn't reflect any specific features we can look for to understand visitor preferences.
    Step 4: Refine the explanation sentence to be an insightful and generalizable interpretation of the aspect without explicitly mentioning specific racial/ethnic groups.
    Step 5: Combine newly discovered aspects with the ones in your current coding. Again, combine similar aspects and remove aspects that are less insightful.
    Your coding should be comprehensive (cover as many aspects mentioned in the corpus as possible) and orthogonal (try your best to derive codes that overlap the least with each other).
    Keep the number of aspects no more than {max_num_aspects}.
    Strictly structure your answer in this format: aspect: content; aspect: content; ... (Each aspect is a phrase describing the captured feature. Each content is a one-sentence explanation of the aspect.) Do not include redundant sentences.)
    
    **New corpus**
    {corpus}
'''
#Step 4: Refine the explanation sentence for each aspect, to explicitly mention what condition leads to the attraction/repulsion of which racial/ethnic groups. For example, "Nice dishes appeals to visitors who enjoy food, contributing to a diverse visitor composition" needs to be refined as "a diverse visitor composition" is super ambiguous. Please specify something like "Mentions of tacos or empanadas appeal to Hipanics while mentions of noodles or dumplings appeal to Asian" (just an example, not necessarily true).
    
PROMPT_EXTRACTION = '''
    Given your current coding of the previously seen corpus, please extract the coding result in the final step (Step 4), strictly following this format: aspect: content; aspect: content; ... (Each aspect is a phrase describing the captured feature. Each content is a one-sentence explanation of the aspect.)

    **Current coding**
    {current_coding}
'''

def Coding(client, corpus, max_num_aspects, current_coding=None):
    dialogs = []
    dialogs.append(functions.encap_msg(SYSTEM, 'system'))
    if(current_coding):
        prompt = PROMPT_CODING.format(corpus=corpus, max_num_aspects=max_num_aspects, current_coding=current_coding)
    else:
        prompt = PROMPT_CODING_FIRST.format(corpus=corpus, max_num_aspects=max_num_aspects)
    dialogs.append(functions.encap_msg(prompt))
    retry_flag, answer, prompt_cost, completion_cost = functions.get_gpt_completion(dialogs, client, tools=None, tool_choice=None, temperature=0.6, max_tokens=MAX_TOKENS)
    return prompt, answer, prompt_cost, completion_cost

def Extract(client, current_coding):
    dialogs = []
    dialogs.append(functions.encap_msg(SYSTEM, 'system'))
    prompt = PROMPT_EXTRACTION.format(current_coding=current_coding)
    dialogs.append(functions.encap_msg(prompt))
    retry_flag, answer, prompt_cost, completion_cost = functions.get_gpt_completion(dialogs, client, tools=None, tool_choice=None, temperature=0.6, max_tokens=MAX_TOKENS)
    return prompt, answer, prompt_cost, completion_cost

def main(client):
    start_time_total = time.time()
    total_prompt_cost, total_completion_cost = 0, 0

    # Load GPT labeled data (generated by GPT_plus.py)
    file_list = glob.glob('./coding_results/analyze_aggregate_result_*.csv')
    for file_idx in range(len(file_list)):
        this_df = pd.read_csv(file_list[file_idx])
        if(file_idx==0):
            df = this_df.copy()
        else:
            df = pd.concat([df, this_df], ignore_index=True)
    
    df.drop_duplicates(subset=['placekey'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    print('Num of fully labeled POIs: ', len(df))

    max_num_aspects = 10
    chunk_size = 10 #5 #10
    num_chunks = int(np.ceil(len(df) / chunk_size))
    current_coding = None
    coding_df = pd.DataFrame(columns=['round', 'prompt', 'coding_answer', 'current_coding'])
    for chunk_idx in range(num_chunks):
        corpus = []
        if((chunk_idx+1)*chunk_size <= len(df)):
            corpus = list(df.loc[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size-1, 'attribution_answer'])
        else:
            corpus = list(df.loc[chunk_idx*chunk_size : , 'attribution_answer'])

        coding_prompt, coding_answer, prompt_cost, completion_cost = Coding(client, corpus, max_num_aspects, current_coding=current_coding)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost

        extraction_prompt, new_coding, prompt_cost, completion_cost = Extract(client, coding_answer)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost

        current_coding = new_coding
        print(f'\nChunk {chunk_idx}, current_coding:\n{current_coding}')

        coding_df.loc[-1] = [chunk_idx, coding_prompt, coding_answer, current_coding]
        coding_df.index = coding_df.index + 1

    # Save results
    coding_df.reset_index(inplace=True, drop=True)
    savepath = (f'coding_results/coding_df_maxnum{max_num_aspects}_chunksize{chunk_size}.csv')
    coding_df.to_csv(savepath, index=False)
    print('Coding results saved at: ', savepath)

    total_run_time_total = round(time.time()-start_time_total, 3)
    print(f'Total_run_time: {total_run_time_total} s')

    total_prompt_cost *= round(PROMPT_COST_1M / 1e6, 6)
    total_completion_cost *= round(COMPLETION_COST_1M / 1e6, 6)
    print(f'Total token usage: {total_prompt_cost + total_completion_cost} USD')

    pdb.set_trace()

if __name__ == "__main__":
    client_choice = 'azureopenai'
    if client_choice == 'siliconflow':
        API_KEY = 'your key' #siliconflow
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.siliconflow.cn/v1"
        )
        model_name = 'deepseek-ai/DeepSeek-V2.5'
    elif client_choice =='azureopenai':
        API_KEY= 'your key'
        ENDPOINT = 'your url'
        client = AzureOpenAI(
            api_key=API_KEY,  # api key
            api_version="2024-07-01-preview",
            azure_endpoint=ENDPOINT  # end point
        )
        model_name = 'gpt-4o'
    elif client_choice == 'openai':
        API_KEY = 'your key'
        client = OpenAI(api_key=API_KEY, base_url='https://api3.apifans.com/v1')

    main(client)