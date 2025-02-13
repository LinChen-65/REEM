# Code for: Invisible Walls in Cities: Leveraging Large Language Models to Predict Urban Segregation Experience with Social Media Content

## Data-Collection

**Note**:   
The repo provides processed intermediate data, you can choose to skip *1.pre-processed* and use the intermediate data directly.

### 1. pre-processed
1. Download dataset: Considering the permission to use the raw dataset, you need to download the raw dataset according to `data/rawdata/README.md` before running the following code.
2. Run `CBG_demographic_feature.ipynb`: This script is used to collect demographic features for city Community Blocks Groups (CBGs). The collected data is saved in the file `data/{city}/{city}_cbg_features_group.csv`
3. Run `poi_safegraph_feature.ipynb`: This script gathers information about city Points of Interest (POIs), including visitor data, location, and other utilized features. The corresponding results are saved in the following files:   
   - `data/{city}/{city}_poi_visitor.csv` 
   - `data/{city}/{city}_poi_location.csv` 
   - `data/{city}/{city}_poi_features.csv`
4. Run `Yelp_Safegraph_align.ipynb`: This script is responsible for collecting Yelp data for city POIs, aligning POIs' SafeGraph and Yelp data, and obtaining descriptive text for images using LLM. The results are saved in the file  `data/Yelp/{city}_poi_with_yelp_review_image_imagestext_GPT4v.csv` 


### 2. Data-Collection

Run `data_process.ipynb`: This script calculates the racial segregation for POIs and the gap between the visitor proportion and demographic proportion of these POIs. The results of this analysis are saved in two files:     
   - `data/{city}/{city}_2019_segregationindex.csv`
   - `data/{city}/{city}_realseg+population_dif.csv`




## RelfectiveLLMCoder

### Obtain Codebook 
Run ```code/RelfectiveLLMCoder/GPT_coding.py```

### Obtian Rating
Please run 
```python GPT_rating.py  --start_idx xx --end_idx xxx --save_root ./data/ --city target-city --client_choice azureopenai``` 
to obtain the rating results `/data/{city}/rating/rating_result_xxx.csv`

ps: start_idx and end_idx are used to split the data into chunks so that they can be processed synchronously.


## RE`EM

### 1.Organize datasets

Run ```code/RE`EM/extracted-dataset.ipynb```: to organize the datasets used for model training. The results are saved in the```code/RE`EM/dataset```directory.

### 2.Train RE`EM different Compnents

The ```code/RE`EM/model``` directory provides examples of the following training code outputs.

**1. Embedding**

- Download the open-source GTE-base model parameters from [HuggingFace](https://huggingface.co/thenlper/gte-base) and save them under ```code/RE`EM/model_pretrain/```
- Run `python R&E_Embedding_train.py --city target-city` : Finetune the GTE-base model and Embbeding Adapter.
- Run `R&E_Embedding_load.py --city target-city`: Load the finetuned Embedding Component model parameters and obtain POI review representations as input for the subsequent model.

**2. Population**

- Run `R&E_Population.py --city target-city` : Train the Population Adapter models.

**3. Reasoning**
- Run `R&E_Reasoning.py --city target-city`: Train the Reasoning Adapter model.

**4. Neighbor-aware Mulit-view Fison**  
- Run ```GNN-Graph.ipynb```: Collect Neighbor information.
- Run `R&E_GNNFusion.py --city target-city`: Train the Neighbor-aware Multi-view Fusion . For reproducible results, the repository provides the model parameters for this step in ```code/RE`EM/best-model```.
