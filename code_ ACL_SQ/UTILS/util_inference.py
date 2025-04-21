import pandas as pd
from UTILS.util_prompts_llm import *
from dotenv import load_dotenv
from UTILS.util_llms import *
import random

        
def  get_questions(log,arg_text_pred, idx_repetition,idx_essay,args,LLMs,essay_id,eg_list,sch_list) :
        # I. generation of the questions from theo===============================================
        questions_llm = ''
        questions_sch = ''
        sq1 = sq2 = []
        sq_llm = sq_sch = None
        retry_count_gene =retry_count_gene2 =retry_count_choice =0
        if 'G' in args.task_id:
            #1. prompt composition____________
            prompt = prompt_question(essay_segmented= arg_text_pred,nb_questions=args.nb_questions,idx_essay = idx_essay,idx_repetition= idx_repetition,log=log)

            #2. run ________________________              
            llm_output = get_llm_output(LLMs['SQ gene'],prompt,log,idx_essay = idx_essay,idx_repetition= idx_repetition)
            
            #3. Extract ____________________
            _, sq1, retry_count_gene, _,_,_ = extract_llm_prediction(
            llm_output,log,args, LLMs['SQ gene'], prompt,essay_id,type_extraction='sq', output_query='pred_list',)
            for i in range(len(sq1)):
                    sq1[i]['prompt'] = prompt
                    
            if sq1 is None or len(sq1) < 3:
                return None, None,None,None, retry_count_gene
            questions_llm  = ''.join(sq1[i]['cq'] for i in range(len(sq1)))
            
        # II. generation of the questions without theo ==========================================
        if 'H' in args.task_id or 'E' in args.task_id:
            sq2 = []
            prompt = prompt_question_sch(essay_segmented= arg_text_pred,sch_list =sch_list ,nb_questions=args.nb_questions,idx_essay = idx_essay,idx_repetition= idx_repetition,log=log,task = args.task_id)

            #2. run ________________________              
            llm_output = get_llm_output(LLMs['SQ gene'],prompt,log,idx_essay = idx_essay,idx_repetition= idx_repetition)
            
            #3. Extract ____________________
            _, sq2, retry_count_gene2, _,_,_ = extract_llm_prediction(
            llm_output,log,args, LLMs['SQ gene'], prompt,essay_id,type_extraction='sq', output_query='pred_list',)
            retry_count_gene += retry_count_gene2
        
            if sq2 is None or len(sq2) < 3:
                return None, None,None,None,retry_count_gene
            print(len(sq2))    
            questions_sch  = ''.join(sq2[i]['cq'] for i in range(len(sq2))) 
            
        # III. Selection of the best questions===========================================
        # choice the best
        questions=  questions_sch+  questions_llm
        if 'C' in args.task_id :
            prompt = prompt_select_question( essay_segmented= arg_text_pred,task=args.task_id,questions= questions ,idx_essay = idx_essay,idx_repetition= idx_repetition,log=log,sch_list =sch_list)

            #2. run ________________________              
            llm_output = get_llm_output(LLMs['SQ choice'],prompt,log,idx_essay,idx_repetition)
            llm_output = re.sub(r'\?(?!\n)', '?\n ', llm_output)
            #3. Extract ____________________
            _, sq, retry_count_choice, _,_,_ = extract_llm_prediction(
            llm_output,log,args, LLMs['SQ choice'], prompt,essay_id,type_extraction='sq_choice', output_query='pred_list',sq_list_llm =[sq1[i]['cq'] for i in range(len(sq1))],sq_list_sch = [sq2[i]['cq'] for i in range(len(sq2))] )

            #find the kind of question that has been selected
            for i in range(len(sq)):
                sq[i]['prompt'] = prompt

        #no choice
        sq_rand = random.sample(sq1+sq2,3)
        retry_count_choice =0 
        
        return sq,sq1,sq2, sq_rand,retry_count_gene+ retry_count_gene2+retry_count_choice,
    
    
def inference(log,essay,idx_repetition,idx_essay,args,df,LLMs,essay_id,start_time,eg_list_segm,eg_list=None,sch_list=None):
    #get the predicted segmenntation, type and quality of each argument of the essay, and related critical questions
    text_with_label = essay
        
    # GET GENERATED QUESTIONS ___________________________________________________________________________________
    if 'G' in args.task_id or 'H' in args.task_id or 'E' in args.task_id:
        sq, sq_llm,sq_sch,sq_rand,retry_count =\
            get_questions(log,text_with_label, idx_repetition,idx_essay,args,LLMs,essay_id,eg_list,sch_list=sch_list) 
        
        if sq is None: # after max_retries the output is still not correct
            return None, None, None
        
    # STORE #########################################################################################################
    try:
        data_pred_essay = pd.DataFrame({
            'essay_id':[essay_id],
            'essay_text': [essay],
            'duration': [str(round((time.time() - start_time)/60,2))]})     
        
        if args.SQ_bool == True:
            quest_pred_essay = pd.DataFrame({})
            if 'G' in args.task_id or 'H' in args.task_id or 'E' in args.task_id:
                data_pred_essay['sq'] = [sq]# store the questions for data to evaluate with the competition eval mth
                data_pred_essay['sq_llm'] = [sq_llm]
                data_pred_essay['sq_sch'] = [sq_sch]
                data_pred_essay['sq_rand'] = [sq_rand]
                quest_pred_essay['essay_id'] =[essay_id] * len(sq)
                quest_pred_essay['Q_id'] = [item['id'] for item in sq] 
                quest_pred_essay['Q_text'] = [item['cq'] for item in sq] 
        else: 
            quest_pred_essay = None
        df_pred_essay = None

    
    except Exception as e:  
        log('Error in the storage of the prediction')
        return None, None, None
    
    return df_pred_essay,data_pred_essay, quest_pred_essay



  