 
if __name__ == "__main__":
    # 0. INITIALISATION  =============================================================================
    import time
    from DATA.preproc_data import preproc_data
    from UTILS.util_prompts_llm import *
    from UTILS.util import *
    import argparse 
    args_init = argparse.Namespace()
    
    # 1. preproc data and set the list of examples for the prompt
    args_init.preproc_data_bool = True
    args_init.dataset_kind = ''#'_train'
    
    #2. run 
    args_init.run_bool = True
    args_init.SQ_bool = True 

    args_init.name_dataset = 'sq'
    args_init.nb_repetition = 1
    args_init.nb_select_data = 1
    args_init.task_id ='G_E_C' #G -> generation, H or E -> type of the template for the argumentative schemes, C-> judge
    
    args_init.run_type_list = ['no']
    args_init.nb_shot = 0 

    #3. AM
    args_init.model_AM_segm = None
    args_init.model_AM = None
    # 4. SQ
    args_init.model_SQ_gene ='llama3.1'
    args_init.model_SQ_choice ='gemma2'
    args_init.nb_questions = 4
    args_init.spe_setting_name =['']
    
    #5.ana 
    args_init.name_list = []
    args_init = set_n_check_args_init(args_init)
    start_global_time = time.time()
    
    # 1. PREPROC=============================================================================
    if args_init.preproc_data_bool: # preproc the data: #set and save data.csv and df.csv
        preproc_data(args_init.name_dataset)
        

    # 2. RUNS (AND ANALYSIS OF THE RUNS)================================================
    if  args_init.run_bool:
        print("gpu available:" +str(torch.cuda.is_available())) 
        
        # execute all the run from: the list run_type_list
        for idx_run, run_type in enumerate(args_init.run_type_list):
            print('\033[35m RUN ' + str(idx_run)+ '/'+ str(len(args_init.run_type_list))+ ' :  '+run_type + '\033[0m')    
            args = set_run(args_init,run_type)
             
    print('END. Total duration: '+str(round((time.time() - start_global_time)/60,2)) )  
