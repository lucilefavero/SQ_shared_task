
import numpy as np
import os
import re
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import pytz
import os
import time
from datetime import datetime
import warnings
from colorama import Fore, Style
import random
import ollama
from tabulate import tabulate
from pprint import pprint
warnings.simplefilter(action='ignore', category=FutureWarning)

def set_n_check_args_init(args_init):
    def message_and_exit(message, color_code='\033[33m'):
        # Remove the condition from the function - it should always execute when called
        reset = '\033[0m'
        print(f"{color_code}{message}{reset}")
        os._exit(0)  # This will immediately terminate the program
        
    #less modify parameters___________________________________________________________
    args_init.data_kind = 'test' #train, val, dev, test will use the essay from this split
    args_init.show_bool = True #log the content of the errors
    args_init.eg_prompt_id =''
    # Set the seed for reproducibility_________________________________________________
    args_init.seed = [i for i in range(args_init.nb_repetition)]
    
    #Grab the run to analyse_____________________________________________________________    
    if args_init.run_bool:
        args_init.name_list = ['']  
             
    elif args_init.name_list == 'temp' or args_init.name_list == 'keep':
        # Define the starting directory
        base_dir = "RESULT/"+ args_init.name_list
        matching_folders = []
        try:
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    # Check if the folder name starts with a 14-digit number
                    if dir_name[:14].isdigit() and len(dir_name) >= 14 and any(keyword in dir_name for keyword in args_init.spe_setting_name):
                        full_path = os.path.join(root, dir_name)
                        matching_folders.append(dir_name)

            # Print or process the matching folders
            for folder in matching_folders:
                print(folder)
            args_init.name_list = matching_folders 
        except Exception as e:
            print(f'Error when load file to analyse: {e}')
        
    #Check validity of models
    models = ollama.list()
    valid_models = {model['model'] for model in models['models']}
    model_attributes = ['model_AM_segm', 'model_AM', 'model_SQ_gene', 'model_SQ_choice']
    for attr in model_attributes:
        model_name = getattr(args_init, attr, None)
        if model_name and 'gpt' not in model_name:  # Only check if model_name exists
            is_substring = any(model_name in valid_model
                for valid_model in valid_models)
            if not is_substring:
                print(f"Invalid model: {model_name}")
                models = ollama.list()
                headers = ["Model Name", "Size", "Modified"]
                table = [[model['model'], model['size'], model['modified_at']] for model in models['models']]
                print(tabulate(table, headers, tablefmt="grid"))
                message_and_exit(f"'{model_name}' is not a valid model.")

    #print the parameters____________________________________________________________________
    pprint(vars(args_init))
    return args_init    
        
           
def jupi(args_init): #set the default values of args
    args = args_init
    #setting of the run ____________________________________________________________________
    args.save_bool = True #save 
    args.max_retries = 7 #number of retries when the llm does not return correct format 
    args.total_duration = []
    args.failure_count = []
    args.selected_essay =[] 
    #task and prompt ______________________________________________________________________
    args.prompt_essay_bool  = False #if True, the prompt of the essay is given in the prompt
    
    #directories_____________________________________________________________________________
    args.dir_data_proc = './DATA/proc/'+  args.name_dataset + '/'
    if args.name_dataset  == 'fb':
        args.dir_save = './RESULT/temp/'+  args.name_dataset + '/'+args.task_id+'/'+args.model_AM.replace('/','_')        
    elif args.name_dataset  == 'sq':
        args.dir_save = './RESULT/temp/'+  args.name_dataset + '/'+args.task_id+'/'+args.model_SQ_gene.replace('/','_')
    return args

def jupi_set(args): # FURTHER SETTING
    
    #set the name of the run: args.setting and args.name _____________________________________________________________________________________    
    cor_id = ''#'C' if args.cor_text_bool else ''
    
    # Set model ID name
    def clean_model_name(model_name):
        return model_name.replace('/', '_').replace(':', '_') if model_name else '_'

    #model_id_name_AM = clean_model_name(args.model_AM)
    model_id_name_SQ = clean_model_name(args.model_SQ_gene)
    model_id_name_SQ_choice = clean_model_name(args.model_SQ_choice)
    model_id_name = model_id_name_SQ +model_id_name_SQ_choice
    
    # set setting name
    args.setting = model_id_name+'_R'+str(args.nb_repetition) +'_'+args.name_dataset+\
        cor_id+str(args.nb_select_data) + '_'+args.task_id + '_EG'+str(args.nb_shot)+'_Q'+str(args.nb_questions) # '_EG'+str(args.nb_shot)+'_'+args.eg_prompt_id 
        
    #grab the date and hour to name the run
    timezone = pytz.timezone('CET')
    now = datetime.now(timezone)
    date_hour_str = now.strftime('%Y%m%d%H%M%S')# Format as a string 'YYYY-MM-DD HH'
    # set run name
    # if args.llm_segm is not None:
    #     args.spe_setting_name = args.spe_setting_name + args.llm_segm
    args.name = date_hour_str+'_'+args.setting+args.spe_setting_name[0] 
        
    #create folder to sort the results of this run________________________________________________________________
    args.dir_save = args.dir_save +'/'+ args.name + '/'
    os.makedirs(args.dir_save, exist_ok=True)
    
    return args

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def helper_fct_save(log=None,args=None,to_save=None,name_to_save='test', kind='data',spe_path =False, paper_bool = False):# kind='prediction', result figure
    if args is not None and args.save_bool or args is None:
        if kind == 'data':
            if spe_path:
                to_save.to_csv(spe_path+'/'+name_to_save+'.csv', index=False)
            else:
                to_save.to_csv(args.dir_save+args.name+'_'+name_to_save+'.csv', index=False)
            if log is not None:
                log('saved '+ name_to_save) 
            
        elif kind == 'fig':
            import matplotlib.pyplot as plt
            if spe_path:
                to_save.savefig(spe_path+'/'+name_to_save+'.png')
            else:
                to_save.savefig(args.dir_save + args.name + '_'+name_to_save+'.png')
            if args is not None  and args.show_bool == False:
                plt.close(to_save) 
            #to_save.savefig(args.dir_save + args.name + '_'+name_to_save+'.eps',format='eps')
            if log is not None:
                log('saved '+ name_to_save) 
        else:
            if log is not None:
                log('not saved '+ name_to_save)
    else:
        if log is not None:
            log('not saved '+ name_to_save)
    #if args.save_bool:
    
def set_run(args_init,run_type):
    import RUN
     
    #no a specific run, change the parameters here __________________________________________
    if run_type == 'no': 
        args = jupi(args_init)
        args.nb_shot = args_init.nb_shot
        args = RUN.main(args)
    # end if run_type == 'no':  
    return args

class Log:
    def __init__(self,log_path,time_key=True):
        self.path = log_path
        #if time_key:
            #self.path = self.path.replace('.','{}.'.format(time.strftime('_%Y%m%d%H%M%S',time.localtime(time.time()))))
        if len(self.path) > 255:  # Ensure it's valid and within length limit
            self.path = self.path[:255]
        print(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())),file=open(self.path,'a+'))
        print('log path:', self.path)
        print('************************************************************************',file=open(self.path,'a+'))
    
    def __call__(self,content,show_bool=True):
        t1 = time.strftime('%H:%M:%S',time.localtime(time.time()))
        if show_bool:
            print(Fore.BLUE,content,Style.RESET_ALL)
        #clean the content
        #remove /n
        content = content.replace('\n','')
        print(t1 + content,file=open(self.path,'a+'))
        
    def error_log(self,type,content,show_bool= True, bool_print=True):
        t1 = time.strftime('%H:%M:%S',time.localtime(time.time()))
        if show_bool:
            print(Fore.RED,type,Style.RESET_ALL)
            if bool_print:
                print('\n\n'+content+'\n')
        #clean the content
        #remove /n
        content = content.replace('\n','')
        print("_______________________________________________________________________",file=open(self.path,'a+')) 
        print(t1 + type,file=open(self.path,'a+'))
        if bool_print:      
            print(content,file=open(self.path,'a+'))
        print("_______________________________________________________________________",file=open(self.path,'a+')) 
        

def search_folder(target_folder, search_root):
    """Search for a folder named target_folder inside search_root and its subdirectories."""
    for root, dirs, files in os.walk(search_root):
        if target_folder in dirs:
            folder_path = os.path.join(root, target_folder)
            print(f"Found: {folder_path}")
            return folder_path#[0]  # Returns the first match
    print("Folder not found.")
    return None


  