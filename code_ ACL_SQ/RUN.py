
#from util_model import *
from UTILS.util import *
from UTILS.util_prompts_llm import *
from UTILS.util_inference import *
from datetime import datetime
import logging
import pandas as pd
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)  # Suppress specific warnings
import pickle
import time
timezone = pytz.timezone('CET')

def main(args):
    # I. INITIALIZATION TASK ###############################################################################
    start_global_time = time.time()
    failure_count = 0 # count the number of essay that did not work accrose the repetition
    
    # set further arguments in args and set log__________________________________________________________
    args = jupi_set(args)
    log = Log(args.dir_save+args.name+".log", time_key=True)
    log('NAME OF THE RUN: '+args.name)
    log(str(args))
    
    #II. RUN PREDICTION ##########################################################################################
    for idx_repetition in range(args.nb_repetition):
        
        log('************************************************************************')
        log('Repetition number: '+str(idx_repetition))
        
        # 1. initialisation prediction dataframes_____________________________________________________________ 
        df_pred =  pd.DataFrame({'essay_id': [],'para_id': [], 'arg_id': [],'arg_text': [],
                                 'arg_type': [],'arg_relation':[],'arg_relation_type':[],'arg_eff':[],'overlap':[]})
        data_pred =  pd.DataFrame({'essay_id': [],'duration': [],'retry_segm': [] ,'retry_type': [], 'retry_eff':[],'retry_link': [],
                                    'miss_segm': [],'miss_type': [],'miss_link': [],'miss_eff':[],
                                   'add_segm': [],'add_type': [],'add_link': [],'add_eff':[], })
        quest_pred =  pd.DataFrame({'essay_id': [],'Q_id': [],'Q_text': [],
                                 'Q_type': [],'Q_eff':[],'simi':[]})
        
        # 2. set the seed____________________________________________________________________________________
        seed_everything(args.seed[idx_repetition])
        
        # 3. load and grab a sample______________________________
        #data = pd.read_csv(args.dir_data_proc+f'data_compet.csv') #load data at the level of the essay
        data = pd.read_csv(args.dir_data_proc+f'data{args.dataset_kind}.csv') #f'data_compet.csv') #load data at the level of the essay
        data = data.sample(args.nb_select_data)
        df = None

        # 4. load eg dataset if needed_____________________________________
        eg_list_segm = []
        eg_list = []
                
        # 5. set the LLMs  ______________________________ 
        LLMs = set_llm_models(args,log,idx_repetition)
        
        # 6. loop over the essay__________________________________________________________________________________
        for idx_essay, essay_id in enumerate(data['essay_id']):#loop over essay
            
            # a. init_______________________________________________________________________________________________
            start_time = time.time() #start the time of this essay 
            now = datetime.now(timezone)
            log( now.strftime('%Hh %M min %Ss')+ '  '+args.name + '   '+ str(idx_repetition) + '   essay #'+ str(idx_essay)+ '      ID: '+str(essay_id))
            
            # b. Prediction task segmentation type and generate critical question_____________________________________________________________________   
            essay = data.loc[data['essay_id'] == essay_id, 'essay_text'].values[0]
            sch_list = ast.literal_eval(data.loc[data['essay_id'] == essay_id, 'schemes'].values[0])
            df_pred_essay, data_pred_essay, quest_pred_essay = inference(log,essay,idx_repetition,idx_essay,args,df,LLMs, essay_id,start_time,eg_list_segm,sch_list=sch_list)
            
            if  data_pred_essay is None or quest_pred_essay is None: 
                log.error_log('No pred for essay '+essay_id +  ' duration: '\
                    +str(round((time.time() - start_time)/60,2))+' min',show_bool =True,content='')
                failure_count += 1
 
            else:
                df_pred = pd.concat([df_pred,df_pred_essay])
                data_pred_essay['dataset'] = data.loc[data['essay_id'] == essay_id, 'dataset_id'].values[0]
                data_pred = pd.concat([data_pred,data_pred_essay])
                quest_pred = pd.concat([quest_pred, quest_pred_essay])
                log('Duration of the prediction: '+data_pred_essay['duration'][0]+' min')
                
            #end for i, essay_id in enumerate(data['essay_id']): 

        #end for i, essay_id in enumerate(data['essay_id']):
        
        # 7. Post processing of this repetition______________________________________________________________________________________
        #save the number of failure
        # if failure_count  !=len(args.selected_essay)-len(data_pred['essay_id'].unique()):
        #     raise ValueError('mismatch between the number of essay stored and the number of essay that did not work!')
        log('Number of essay that did not work: '+str(len(args.selected_essay)-len(data_pred['essay_id'].unique()))
            + ' over '+str(len(args.selected_essay))+ ' essays selected')
        args.failure_count.append(failure_count)
        args.total_duration.append(round((time.time() - start_global_time)/60,2))
               
        helper_fct_save(log,args,data_pred,'data_'+str(idx_repetition),kind = 'data')
        helper_fct_save(log,args,quest_pred,'SQ_df_raw_'+str(idx_repetition),kind = 'data')
        with open(args.dir_save+'args'+args.name+'.pkl', 'wb') as f:
            pickle.dump(args, f)
              
        log('End of the repetition: failure count: '+str(failure_count))
    #end for idx_repetition in range(args.nb_repetition):

    #III. SAVE THE RESULTS ###############################################################################
        
    #compute the average duration and failure count
    #args.failure_count = sum(args.failure_count)/args.nb_repetition
    #args.total_duration = sum(args.total_duration)/args.nb_repetition

    # Save the arguments to a pickle file with the updated args.failure_count and args.total_duration
    with open(args.dir_save+'args'+args.name+'.pkl', 'wb') as f: #save the selected set of data, duration and failure count
        pickle.dump(args, f)
        
    log('End of the run '+args.name + 'Duration: '+ str(args.total_duration) +
        'min. Number of essay that did not work: '+str(args.failure_count))
    
    return args
    
pass