import re
import pandas as pd
import json
from UTILS.util_prompts_llm import *
from langchain_ollama import OllamaLLM
from openai import OpenAI
from dotenv import load_dotenv
import string
import difflib
def count_differences(original, modified):
    matcher = difflib.SequenceMatcher(None, original, modified)
    missing_count = sum(i2 - i1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ('replace', 'delete'))
    added_count = sum(j2 - j1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ('replace', 'insert'))
    return missing_count, added_count


def align_text(original, modified):
    # Create a SequenceMatcher
    matcher = difflib.SequenceMatcher(None, original, modified)

    missing_count = 0  # How many words from 'original' are missing in 'modified'
    added_count   = 0  # How many words are newly added in 'modified'

    # These lists will store an aligned version for visualization
    replace_original = []
    replace_modified  = []
    del_from_original = []
    add_in_modified = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Words in original[i1:i2] are replaced by words in modified[j1:j2].
            # This can be thought of as (i2-i1) deletions + (j2-j1) insertions.
            missing_count += (i2 - i1)
            added_count   += (j2 - j1)
            replace_original.extend(original[i1:i2])
            replace_modified.extend(modified[j1:j2])

        elif tag == 'delete':
            # Words in original[i1:i2] are missing in 'modified'
            missing_count += (i2 - i1)
            del_from_original.extend(original[i1:i2])
        elif tag == 'insert':
            # Words in modified[j1:j2] are newly inserted
            added_count += (j2 - j1)
            add_in_modified.extend(modified[j1:j2])
            
    #end for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    # print("Words missing (deleted) from 'modified':", missing_count)
    # print("Words added in 'modified':", added_count)
    
    return replace_original, replace_modified, del_from_original, add_in_modified ,missing_count,added_count

def structure_output(whole_text):
    cqs_list = whole_text.split('\n')
    final = []
    valid = []
    not_valid = []
    for cq in cqs_list:
        if re.match('.*\?(\")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)

    still_not_valid = []
    for text in not_valid:
        new_cqs = re.split("\?\"", text+'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq+'?\"')
        else:
            still_not_valid.append(text)

    for i, cq in enumerate(valid):
        occurrence = re.search(r'[A-Z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])
        else:
            continue

    output = []
    try:
        for i in range(len(final)):
            output.append({'id':i, 'cq':final[i]})
        return output
    except:
        raise ValueError('The output is not in the right format')

def set_llm_models(args, log, idx_repetition):
    
    # Define LLM configurations
    LLMs = {'segm':{'model_name': args.model_AM_segm,}, 
            'AM':{'model_name': args.model_AM,}, 
            'SQ gene':{'model_name': args.model_SQ_gene,}, 
            'SQ choice':{'model_name': args.model_SQ_choice,}}

    # Initialize all LLMs
    for key, value in LLMs.items():
        if value['model_name'] is None:
            LLMs[key]['model'] = None
            LLMs[key]['kind'] = None
        elif 'gpt' in value['model_name'].lower() or 'mini' in value['model_name'].lower() :
            load_dotenv()
            LLMs[key]['model'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            LLMs[key]['kind'] = 'gpt'   
            log(f"{key}: {value['model_name']} set (GPT)")
            
        else:
            LLMs[key]['model']  = OllamaLLM(model=value['model_name'],keep_alive=0, seed=args.seed[idx_repetition])
            LLMs[key]['kind'] = 'ollama'
            log(f"{key}: {value['model_name']}  set (Ollama)")
    #end for key, value in LLMs.items():    
    
    return LLMs


def get_llm_output(llm,prompt,log=None,idx_essay=None, idx_repetition=None,idx_segment=None):
    """
    Retrieve output from a specified large language model (LLM) based on the provided prompt.
    """
    
    llm_output = ''
    
    #GPT        
    if llm['kind'] == 'gpt':
        completion = llm['model'].chat.completions.create(
            model=llm['model_name'],messages=[{ "role": "user","content": prompt}])
        llm_output = completion.choices[0].message.content
    #OLLAMA
    else: 
        llm_output =  llm['model'].invoke(prompt)
    if llm_output=='':
        raise ValueError("No valid LLM output.")

    # Logging the LLM output for the first essay and repetition_______________________________
    if log and idx_essay == 0 and idx_repetition == 0 and (idx_segment == 0 or idx_segment is None):
        log(f"{llm['model_name']} output:\n{llm_output}")
 
    return llm_output

def extract_llm_prediction(llm_output,log,args, llm, prompt,essay_id, 
                           type_extraction= 'essay', essay = None, output_query=None,segm_bool =False, sq_list_llm=False,sq_list_sch=False):
    
    def extract_target(llm_output,target_labels):
        json_part_match = re.search(r"\{[^}]+\}", llm_output, re.DOTALL)  # Separate explanation from evaluation measures
        json_string = json_part_match.group(0)
        # Regular expression to match unquoted words in values (not keys)
        pattern = r'(?<=\[|,)\s*([A-Za-z0-9_]+)\s*(?=,|\])'
        # Replace unquoted words with quoted words
        json_string = re.sub(pattern, r'"\1"', json_string)
        json_string = re.sub(r'\\', '', json_string)
        json_data = json.loads(json_string)

        #if the output got > or < at the beginning or the end remove them
        if json_data[target_labels][0] =='<':
            json_data[target_labels] = json_data[target_labels][1:]
        if json_data[target_labels][-1] =='>':
            json_data[target_labels] = json_data[target_labels][:-1]
        
        segement_target = json_data[target_labels]#.title() #ensure the first letter is capital and the rest is lower 
        return segement_target
        
    def find_labels_with_preceding_text(text_segmented, segment,threshold=0.8):
        #check if segment and text_segmented are not empty
        if not segment.strip() or not text_segmented.strip():
            return None, None, None
        segment_splitted = segment.split()
        list_labels = list(re.finditer(r"<([^>]+)>", text_segmented))
        
        for i in range(len(list_labels)):
            # Get the current label
            label = list_labels[i + 1].group(1)  
            # Get the text between the current < > and the next < >
            preceding_text = text_segmented[list_labels[i].end():list_labels[i + 1].start()].strip()
            
            # Count the elements of 'segment_splitted' present in the preceding text
            elements_present = sum(1 for elem in segment_splitted if elem in preceding_text)
            
            # Check if at least 80% of 'segment_splitted' are in the preceding text
            if elements_present / len(preceding_text.split()) >= threshold: 
                #print(elements_present / len(segment_splitted))
                remaining_text = text_segmented[list_labels[i + 1].start():].strip()
                break

        return label, preceding_text, remaining_text

    def check_generated_essay_equal_to_true_essay(essay, llm_output, threshold = 0.1):
        import string
        original = re.sub(r'<.*?>', ' ', essay.lower())
        modified = re.sub(r'<.*?>', ' ', llm_output.lower())
        translator = str.maketrans('', '', string.punctuation)
        original = original.translate(translator)
        modified = modified.translate(translator)
        missing_count, added_count= count_differences(original.split(), modified.split())
        replace_original, replace_modified, del_from_original,\
            add_in_modified ,missing_count,added_count = align_text(original.split(),  modified.split())
        missing_percentage = missing_count/len(original.split())

        if missing_percentage > threshold:
            raise ValueError(f"Predicted text contains 'too many missing and added words: {missing_count} words missing, {added_count} words added")
        
        return missing_count,added_count, missing_percentage


    def output_error_handling(log, e,args,llm, prompt,essay_id, retry_count,llm_output,segm_bool):
        retry_count += 1  # Increment retry counter
        log.error_log(f'Error in essay {essay_id}, retry {retry_count}/{args.max_retries}. Error: {e}',llm_output,args.show_bool)
        
        if retry_count < args.max_retries: 
            llm_output = get_llm_output(llm, prompt,log,0,0) #will display the prompt
            return llm_output, retry_count
        
        else: # if don't succeed after max_retries ignore this segment
            log.error_log(f"\033[31m Not able to perform the task on this argument (essay {essay_id}) after \033[0m"+str(retry_count)+"\033[31m retries \033[0m",'',args.show_bool)
            log_err  = pd.DataFrame({'name': args.name ,'essay_id': essay_id, 'error': [str(e)],'prompt': prompt, 'llm_output': llm_output,'llm': [llm]} )
            df = pd.read_csv(f'log_err.csv')
            df = pd.concat([df,log_err], ignore_index=True)
            df.to_csv(f'log_err.csv', index=False)
            log("saved error in log_err.csv")

            return None, retry_count
    
    def extract_essay(llm_output,  essay, output_query):
        llm_output=llm_output.split('</think>', 1)[-1].strip()
        # check if generated essay is equal to the true essay   
        pred_list = []
        #clean the output 
        llm_output=llm_output.replace('#ESSAY:', '')
        if llm_output.endswith("#OUTPUT:"):
            llm_output=llm_output.replace('#OUTPUT:', '')
        if llm_output.endswith("#OUTPUT:<SEP>"):
            llm_output=llm_output.replace('#OUTPUT:<SEP>', '')
        if llm_output.endswith("#Output:"):
            llm_output=llm_output.replace('#Output:', '')
        else:
            llm_output = llm_output.split("#OUTPUT:")[-1].strip()
        if len(llm_output.split('#')) >2:
            llm_output=llm_output.replace('#Ineffective:', '')
        llm_output = llm_output.split('#Ineffective:')[-1].strip()                        
        llm_output = llm_output.replace('\n', ' ')
        pattern = r"^(.*<.*>)"
        match = re.search(pattern, llm_output, re.DOTALL)
        llm_output = match.group(1)
        llm_output = llm_output.replace('<SEP><SEP>', ' ')
        # check the similarity between the predicted and the true essay when performing the segmentation, raise error if lower than a threshold
        miss, add, missing_percentage = check_generated_essay_equal_to_true_essay(essay, llm_output, threshold = 0.1)

        #extract the segments
        segment_list = re.split(r'<[^>]*>', llm_output)
        segment_list= segment_list[:-1] #remove the last empty strings
        
        # extract the target
        if output_query == 'pred_list':    
            text = '<SEP>'+llm_output
            for segment in segment_list:
                label, segment_predic, remaining_text = find_labels_with_preceding_text(text, segment,threshold=0.7)
                text = remaining_text
                pred_list.append(label)
            #end for segment in essay_segmented.split("<SEP>"):
   
            if len(pred_list) != len(segment_list):
                raise Exception('Different number of type of argument component: generated: '
                        +len(pred_list)+'true: '+len(segment_list))
                
            pred_list = [None if item is None else item.split(',') for item in pred_list]
            
        success = True
        return success, segment_list, pred_list, success, miss, add, missing_percentage
    
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #output_query='pred_list'
    retry_count = 0
    success = False  # Flag to track if inference is successful 
    
    while retry_count < args.max_retries and not success:
        try:
            if type_extraction == 'essay':
                success, segment_list, pred_list, success, miss, add, missing_percentage = extract_essay(llm_output,  essay, output_query)
                
                
            elif type_extraction == 'dico': #extract the output as a dictionary
                #a. extract the target (target_labels is either TYPE, QUALITY or TYPE AND QUALITY...)
                pred_list= extract_target(llm_output,args.target_name) 
                segment_list = [] 
                miss = add = missing_percentage = None
                success = True
                
            elif type_extraction == 'sq': 
                llm_output=llm_output.replace('#CRITICAL QUESTIONS:','')
                llm_output=llm_output.split('</think>', 1)[-1].strip()
                pred_list = structure_output(llm_output)
                segment_list = [] 
                miss = add = missing_percentage = None
                
                if len(pred_list) !=  args.nb_questions:
                    print(f"Invalid number of critical questions: {len(pred_list)} instead of {args.nb_questions}")
                    #raise ValueError(f"Invalid number of critical questions: {len(pred_list)} instead of {args.nb_questions}")
                success = True
                
            elif type_extraction == 'sq_choice': 
                llm_output=llm_output.split('</think>', 1)[-1].strip()
                llm_output=llm_output.split('#CRITICAL QUESTIONS:', 1)[-1].strip()
                #llm_output = llm_output.replace("?", "? \n ")
                pred_list = structure_output(llm_output)
                
                #check validity output____________________________________________________________________________________
                # are the critical questions in the list of possible questions
                for idx_question in range(len(pred_list)):
                    pred_list[idx_question]['kind'] = ''
                    idx_true_question = 0
                    missing_percentage = 1
                    stopping_criterion_llm = -1
                    stopping_criterion_sch = -1
                    if 'G' in args.task_id:
                        stopping_criterion_llm = len(sq_list_llm)
                    if 'H' in args.task_id or 'E' in args.task_id:
                        stopping_criterion_sch = len(sq_list_sch)
                    while missing_percentage > 0.1 and idx_true_question <stopping_criterion_llm:# and idx_true_question <=len(sq_list_sch) :
                        import string
                        original = re.sub(r'<.*?>', ' ', pred_list[idx_question]['cq'].lower())
                        modified = re.sub(r'<.*?>', ' ', sq_list_llm[idx_true_question].lower())
                        translator = str.maketrans('', '', string.punctuation)
                        original = original.translate(translator)
                        modified = modified.translate(translator)
                        miss, add= count_differences(original.split(), modified.split())
                        replace_original, replace_modified, del_from_original,\
                            add_in_modified ,miss,add = align_text(original.split(),  modified.split())
                        missing_percentage = miss/len(original.split())
                        if missing_percentage < 0.1:
                            pred_list[idx_question]['kind'] = 'llm'
                            break
                        #print(missing_percentage)
                        idx_true_question += 1
                    idx_true_question = 0
                    missing_percentage = 1
                    while missing_percentage > 0.1 and idx_true_question <stopping_criterion_sch:# and idx_true_question <=len(sq_list_sch) :
                        import string

                        original = re.sub(r'<.*?>', ' ', pred_list[idx_question]['cq'].lower())
                        modified = re.sub(r'<.*?>', ' ', sq_list_sch[idx_true_question].lower())
                        translator = str.maketrans('', '', string.punctuation)
                        original = original.translate(translator)
                        modified = modified.translate(translator)
                        miss, add= count_differences(original.split(), modified.split())
                        replace_original, replace_modified, del_from_original,\
                            add_in_modified ,miss,add = align_text(original.split(),  modified.split())
                        missing_percentage = miss/len(original.split())
                        if missing_percentage < 0.1:
                            pred_list[idx_question]['kind'] = 'sch'
                            break
                        #print(missing_percentage)
                        idx_true_question += 1
                    
                if idx_true_question>stopping_criterion_llm and idx_true_question>stopping_criterion_sch :
                    raise ValueError(f"Invalid critical question: '{pred_list[idx_question]['cq']}' not in genrated questions")
                # else:
                #     print(f" {miss} words missing, {add} words added")
                        
                if len(pred_list) !=  3: #at the end we want to select 3 questions
                    #print(f"Invalid number of critical questions: {len(pred_list)} instead of {args.nb_questions}")
                    raise ValueError(f"Invalid number of critical questions: {len(pred_list)} instead of 3")
                #_____________________________________________________________________________________________________
                
                segment_list = [] 
                success = True
            else: 
                raise ValueError("Invalid type_extraction.")
            #end if type_extraction == 'essay':
        except Exception as e:# if error in the output, retry
            llm_output, retry_count = output_error_handling(log, e,args,llm, prompt,essay_id, retry_count,llm_output,segm_bool)
            if llm_output is None:
                return None, None, retry_count, None, None, None
        #end try except

    #end while retry_count < args.max_retries and not success:
    return segment_list, pred_list, retry_count, add, miss, missing_percentage