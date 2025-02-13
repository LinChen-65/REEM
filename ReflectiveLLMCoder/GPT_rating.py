import argparse
import random
import sys
import re
sys.path.append("..")

from openai import OpenAI, AzureOpenAI
import os
import numpy as np
import time
import pandas as pd
import pdb
import ast
import functions
#import tiktoken


PROMPT_COST_1M = 0.15
COMPLETION_COST_1M = 0.6
MODEL_NAME = "gpt-4o" 


#model_token_limit = 128000 # Define the model's token limit (e.g., GPT-4 8k model)
max_response_tokens = 1200 # Set a limit for the output tokens
#available_input_tokens = model_token_limit - max_response_tokens # Calculate available tokens for the input
#encoding = tiktoken.get_encoding("cl100k_base") # Choose the encoding for the model (e.g., "cl100k_base" for GPT-4)



codebook = """
    1. Cultural Resonance and Appeal: Culturally themed offerings, such as Italian-American or South Indian cuisine, attract visitors seeking authentic or familiar experiences, influencing visitation based on cultural representation and resonance;
    2. Price Sensitivity and Economic Accessibility: Moderate pricing, coupons, and cost-effective policies like BYOB appeal to budget-conscious visitors, impacting visitation patterns based on affordability and economic considerations;
    3. Service Quality and Customer Experience: Professional and attentive service, despite occasional inconsistencies, attracts visitors valuing high service standards and personal interactions, influencing demographics based on service expectations;
    4. Atmosphere and Social Environment: Lively, trendy, or family-friendly settings attract visitors prioritizing social and communal experiences, impacting visitation based on social and family-oriented preferences;
    5. Accessibility and Convenience: Central locations, parking availability, and delivery services attract visitors prioritizing efficiency and accessibility, influencing patterns based on transportation and convenience;
    6. Visual and Aesthetic Appeal: Modern, chic, and historically themed environments attract visitors who appreciate aesthetic and immersive experiences, influencing demographics based on visual and cultural preferences;
    7. Cultural and Social Inclusivity: Inclusive, diverse, and culturally sensitive environments attract a broad demographic by catering to varied identities and preferences, influencing visitor composition based on inclusivity and cultural representation;
    8. Product Variety and Quality: Diverse and high-quality offerings, including visually appealing and culturally themed products, attract visitors prioritizing variety and quality, influencing visitation based on product expectations;
    9. Community Engagement and Local Involvement: Establishments with strong community ties and neighborhood vibes attract visitors valuing local engagement and communal experiences, influencing demographics based on community integration and involvement. 
"""


SYSTEM = """
    As a human mobility and computational social science expert, your task is to understand the visitation preference patterns of visitors from different racial/ethnic backgrounds to a specific place (POI), by jointly considering features including its name, location, provided services, visual features, and visitor comments. 
"""

PROMPT_RATING = """
    You are provided with the Yelp Data of a POI named "{name}" in {city}, and a Codebook to guide your analysis dimensions.
    Your objective is to assess the POI's attractiveness to different racial/ethnic groups along 9 dimensions specified in the Codebook. 
    There are 5 racial/ethnic groups: Hispanic, black, asian, white, and other.

    Step 1: Codebook-guided Content Analysis.
    Based on the given Yelp Data, summarize the POI's characteristics along 9 dimensions specified in the Codebook, focusing on how it may attract/repel certain racial/ethnic groups.
    If you are certain, explicitly point out the attraction/repulsion of certain racial/ethnic groups along each dimension.
    Strictly format your analysis as follows: {{'1. Cultural Resonance and Appeal': one-sentence reasoning and summary, '2. Price Sensitivity and Economic Accessibility': one-sentence reasoning and summary, ...}}.
    
    Step 2: Rating.
    Based on the content analysis in Step 1, imagine yourself as member of one of the 5 racial/ethnic groups, and rate the POI along 9 dimensions specified in the Codebook. 
    All ratings range from 0 to 10, where 0 indicates least attraction (strongest repulsion), 5 indicates perfect neutrality (neither attraction nor repulsion), and 10 indicates strongest attraction. 
    Try your best to differentiate the ratings for different racial/ethnic groups based on the content analysis.
    Eventually, you will provide 9 ratings for each racial/ethnic group, resulting in 45 ratings in total.
    Strictly format your assessment as follows: "From the Hispanic perspective {{'1. Cultural Resonance and Appeal':'?','2. Price Sensitivity and Economic Accessibility':'?',...}}. From the Black perspective {{'1. Cultural Resonance and Appeal':'?','2. Price Sensitivity and Economic Accessibility':'?',...}}. ... Explanation: [Elaborate on your reasoning here].
    
    **Input Information**
    Codebook: {codebook}
    Yelp Data: {info}
"""


PROMPT_EXTRACTION = '''
    Extract the answer from the previous response, organizing it strictly into the required format:
    Step 1: {{'1. Cultural Resonance and Appeal': '?', '2. Price Sensitivity and Economic Accessibility': '?', ...}}. 
    Step 2: From the Hispanic perspective {{'1. Cultural Resonance and Appeal':'?','2. Price Sensitivity and Economic Accessibility':'?',...}}. From the Black perspective {{'1. Cultural Resonance and Appeal':'?',...}}. ... Explanation: [Elaborate on your reasoning here].
    
    Criteria: {codebook}
    
    **Previous ANSWER**
    {correction_answer}
'''

from datetime import datetime
def sort_by_date(review):
    return datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S')

##ATTENTION 这里有修改
MAX_REVIEW_NUM = 80 #80
MAX_IMAGE_NUM = 40
SHORT_NUM = 10
SHORT_NUM_IMAGE = 5
def Rating(model_name,client, row, city):
    review_num = row['review_num']
    review_all = list(eval(row['review']))
    sorted_review_all = sorted(review_all, key=sort_by_date, reverse=True) #sort review by time

    image_num = row['images_num']
    if image_num > 0:
        image_list = list(eval(row['images_text']))
    else:
        image_list = []

    attributes_list = [{'categories': row['categories']}, row['attributes']]

    try_flag = 1
    shortlength = -1
    while (try_flag):
        shortlength += 1
        if review_num > (MAX_REVIEW_NUM - shortlength * SHORT_NUM):
            select_review_list = sorted_review_all[:(MAX_REVIEW_NUM - shortlength * SHORT_NUM)]
        else:
            select_review_list = sorted_review_all
        if image_num > (MAX_IMAGE_NUM - shortlength * SHORT_NUM_IMAGE):
            select_image_list = random.sample(image_list, (MAX_IMAGE_NUM - shortlength * SHORT_NUM_IMAGE))
        else:
            select_image_list = image_list

        info_list = attributes_list + select_review_list + select_image_list
        info = ' '.join(str(item) for item in info_list)

        dialogs = []
        dialogs.append(functions.encap_msg(SYSTEM, 'system'))
        prompt = PROMPT_RATING.format(name=row['location_name'], info = info, codebook=codebook, city=city.capitalize())
        dialogs.append(functions.encap_msg(prompt))
        try:
            try_flag, answer, prompt_cost, completion_cost = functions.get_gpt_completion(model_name,dialogs, client, tools=None, tool_choice=None, temperature=0.2, max_tokens=max_response_tokens)
        except:
            continue
    return prompt, answer, prompt_cost, completion_cost


def Extract(model_name,client, correction_answer): #增加一轮整理的prompt
    dialogs = []
    dialogs.append(functions.encap_msg(SYSTEM, 'system'))
    prompt = PROMPT_EXTRACTION.format(correction_answer=correction_answer, codebook=codebook)
    dialogs.append(functions.encap_msg(prompt))
    retry_flag, answer, prompt_cost, completion_cost = functions.get_gpt_completion(model_name,dialogs, client, tools=None, tool_choice=None, temperature=0.2, max_tokens=max_response_tokens)
    return prompt, answer, prompt_cost, completion_cost


def SaveResult(reasoning_result, save_root, city, file_num):
    reasoning_result_df = pd.DataFrame(reasoning_result)
    city_save_root = os.path.join(save_root, city,'rating')
    if(not os.path.exists(city_save_root)): 
        os.makedirs(city_save_root)
    save_path = os.path.join(city_save_root, f'rating_result_{file_num}.csv')
    reasoning_result_df.to_csv(save_path, index=False)
    print('Result saved at: ', save_path)


def ExtractDict(text):
    # Regular expression to match the dictionaries for each racial/ethnic group
    pattern = r"\*{0,2}From the (\w+) perspective\*{0,2}\s*\n*\s*\:*\s*\n*\s*(\{.*?\})\s*\.*"

    # Find all matches for racial/ethnic groups
    try:
        matches = re.findall(pattern, text,re.DOTALL)
    except Exception as e:
        print(f"An error of type {type(e).__name__} occurred: {e}")
        return None
    # Convert the matches to dictionaries
    result_dicts = {}
    for match in matches:
        group = match[0]
        #dict_str = "{" + match[1] + "}"
        dict_str =  match[1]
        # Convert the string to a dictionary and convert the ratings to integers
        ratings_dict = ast.literal_eval(dict_str)
        ratings_dict = {key: int(value) for key, value in ratings_dict.items()}  # Convert values to integers
        result_dicts[group] = ratings_dict
    #print(result_dicts)

    # Reorder result_dicts based on ordered_keys
    try:
        ordered_keys = ['Hispanic', 'Black', 'Asian', 'White', 'Other']
        reordered_result_dicts = {key: result_dicts[key] for key in ordered_keys}
    except:
        try:
            ordered_keys = ['hispanic', 'black', 'asian', 'white', 'other']
            reordered_result_dicts = {key: result_dicts[key] for key in ordered_keys}
        except Exception as e:
            print(f"An error of type {type(e).__name__} occurred: {e}")
            return None
    return reordered_result_dicts

def Dict2List(race_dicts):
    if (race_dicts):
        # Extract all integers into a list
        ratings = [value for group in race_dicts.values() for value in group.values()]
        return 0, ratings
    else:
        return 1, []

def main(start_idx, end_idx, save_root, city, model_name,client,data_path):
    start_time_total = time.time()
    total_prompt_cost, total_completion_cost = 0, 0

    #selected POI feature+seg+dif+summary
    poi_df = pd.read_csv(data_path+f'{city}/{city}_2019_segregationindex.csv') # Philadelphia: (5360, 46)

    image_files = os.listdir(data_path+f'{city}/')
    pattern = f'{city}_poi_with_yelp_review_image_imagestext_'
    csv_files = [file for file in image_files if file.startswith(pattern) and file.endswith('.csv')]
    images_pd = pd.read_csv(data_path+f'{city}/'+csv_files[0]) # usecols=['placekey', 'images_text','attributes'])
    poi_df = poi_df.merge(images_pd, on='placekey', how='inner')
    del images_pd

    poi_df = poi_df.drop_duplicates(subset='placekey', keep='first').reset_index(drop=True)
    columns_to_drop = [col for col in poi_df.columns if
                       'income' in col or '_pre' in col or '_dif' in col or 'Unnamed' in col]
    poi_df = poi_df.drop(columns=columns_to_drop)
    print(f'{city} have total {poi_df.shape[0]} POI ') #with columns {poi_df.columns}
    print(f'max review_num: {poi_df.review_num.max()}, max images_num: {poi_df.images_num.max()}')

    poi_cut_range = list(range(start_idx, end_idx, 1))
    print(f'Waiting to Rate: POI {poi_cut_range[0]} to POI {poi_cut_range[-1]}')
    poi_df = poi_df.iloc[poi_cut_range]
    poi_df['attributes'] = poi_df['attributes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})

    poi_df['prompt_cost'] = 0
    poi_df['completion_cost'] = 0

    reasoning_result = []
    for index, row in poi_df.iterrows():
        start_time = time.time()
        retry_row = 1
        while retry_row:
            # Rate
            rating_prompt, rating_answer, prompt_cost, completion_cost = Rating(model_name, client, row, city)
            row['coderate_prompt'] = rating_prompt
            row['coderate_answer'] = rating_answer
            row['prompt_cost'] += prompt_cost
            row['completion_cost'] += completion_cost

            # Extraction
            extraction_prompt, extraction_answer, prompt_cost, completion_cost = Extract(model_name,client, rating_answer)
            row['extract_answer'] = extraction_answer
            row['prompt_cost'] += prompt_cost
            row['completion_cost'] += completion_cost

            race_dicts = ExtractDict(extraction_answer)
            retry_row_flag,ratings_result = Dict2List(race_dicts)
            if retry_row_flag:
                retry_row +=1
                if retry_row>5:
                    row['rating'] = []
                    print(
                        f'{index} POI Rating fail. Used time: {round(time.time() - start_time, 3)} s.  Prompt Token used: {row["prompt_cost"]}.  Completion Token used: {row["completion_cost"]}')
                    break
            else:
                row['rating'] = ratings_result #list
                print(f'retry time: {retry_row}')
                retry_row = 0
                print(f'{index} POI Rating done. Used time: {round(time.time() - start_time, 3)} s.  Prompt Token used: {row["prompt_cost"]}.  Completion Token used: {row["completion_cost"]}')
        total_prompt_cost += row['prompt_cost']
        total_completion_cost += row['completion_cost']
        reasoning_result.append(row)

        if((index+1)%10==0):
            print(f'{index}, Total_run_time: {round(time.time()-start_time_total, 3)} s')
            SaveResult(reasoning_result, save_root, city, index//10)
            reasoning_result = []

    print(f'Total_run_time: {round(time.time()-start_time_total, 3)} s')
    if len(reasoning_result) > 0:
        SaveResult(reasoning_result, save_root, city, index//10)
    

    total_prompt_cost *= round(PROMPT_COST_1M / 1e6, 6)
    total_completion_cost *= round(COMPLETION_COST_1M / 1e6, 6)
    print(f'Total token usage: {total_prompt_cost + total_completion_cost} USD')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CodeRateGeneration Action.")
    parser.add_argument("--start_idx", type=int, required=True, help="Start index for range.")
    parser.add_argument("--end_idx", type=int, required=True, help="End index for range")
    parser.add_argument("--save_root", type=str,required=True,default='./data/',help='Folder path of saved data and result')#现是全部的
    parser.add_argument("--city", default='Philadelphia')
    parser.add_argument("--client_choice", default='openai')
    args = parser.parse_args()
    assert args.client_choice in ['openai', 'azureopenai','siliconflow']

    ##init client
    if args.client_choice == 'siliconflow':
        API_KEY = 'your key' #siliconflow
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.siliconflow.cn/v1"
        )
        model_name = 'deepseek-ai/DeepSeek-V2.5'
    elif args.client_choice =='azureopenai':
        API_KEY= 'your key'
        ENDPOINT = 'your url'
        client = AzureOpenAI(
            api_key=API_KEY,  # api key
            api_version="2024-07-01-preview",
            azure_endpoint=ENDPOINT  # end point
        )
        model_name = 'gpt-4o-mini'
    elif args.client_choice == 'openai':
        API_KEY = 'your key'
        client = OpenAI(api_key=API_KEY, base_url='https://api3.apifans.com/v1')

    main(args.start_idx, args.end_idx, args.save_root, args.city,model_name, client,args.save_root)

    '''
    MAX_REVIEW = 80
    MAX_IMAGE = 40
    '''

