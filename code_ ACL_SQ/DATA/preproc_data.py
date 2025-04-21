
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def preproc_data(name_dataset):
    # SQ ================================================================================================        
    if name_dataset == 'sq':
        #1. load row data
        print('loading data')
        with open(f'./DATA/row/sq/validation.json','r', encoding='utf-8') as file:
            raw_data = json.load(file)
        
        # 2. grab the relevant data 
        essay_id_list = []
        essay_text_list = []
        sq_list = []
        dataset_id = []
        schemes =[]
        
        for idx_essay in raw_data.keys():
            print(idx_essay)
            essay_id_list.append(idx_essay)
            essay =raw_data[idx_essay]['intervention'].split('"',1)[1]
            essay_text_list.append(essay[:-1])
            sq_list.append(raw_data[idx_essay]['cqs'])
            dataset_id.append(raw_data[idx_essay]['dataset'])
            schemes.append(raw_data[idx_essay]['schemes'])   
            
        # 3. create the DataFrame data
        data = pd.DataFrame({'dataset_id': dataset_id, 'essay_id': essay_id_list, 'essay_text': essay_text_list,
                            'schemes':schemes, 'sq': sq_list,})
        #clean sche
        data['schemes_proc'] = data['schemes'].apply(lambda x: list(dict.fromkeys(x)))
        existing_labels = set([label for sublist in data['schemes_proc'] for label in sublist])

        def merge_er_labels(label_list):
            new_list = []
            for label in label_list:
                if label == "ERAdHominem": 
                    new_list.append("Ad hominem")
                elif label == "ERExpertOpinion": 
                    new_list.append("ExpertOpinion")
                elif label.startswith("ER") and label[2:] in existing_labels:
                    new_list.append(label[2:])
                else:
                    new_list.append(label)
            return new_list

        # Apply the merging function
        data['schemes'] = data['schemes_proc'].apply(merge_er_labels)
        #4. save the data    
        data_train, data_temp = train_test_split(data, test_size=0.6, random_state=3948)
        data_val, data_test = train_test_split(data_temp, test_size=0.7, random_state=3948)

        data_train.to_csv('./DATA/proc/sq/data_train.csv', index=False)
        print('data_train saved. lenght ' +str(len(data_train)))
        data_val.to_csv('./DATA/proc/sq/data_val.csv', index=False)
        print('data_val saved. lenght ' +str(len(data_val)))
        data_test.to_csv('./DATA/proc/sq/data_test.csv', index=False)
        print('data_test saved. lenght ' +str(len(data_test)))
        
    print('data processed')