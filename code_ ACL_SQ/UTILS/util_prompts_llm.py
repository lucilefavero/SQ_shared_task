from UTILS.util import *
import random
import pandas as pd
import ast

def prompt_compo_eg(nb_shot, eg_list):
    if nb_shot == 0:
        eg_prompt = '' 
    else:
        #eg_list = pd.read_csv('./DATA/proc/'+name_dataset+'/eg_for_prompt_'+eg_prompt_id+corrected_id+'.csv')#.iloc[0:nb_shot]
        eg_list = eg_list['0'].tolist()
        eg_prompt_temp = random.sample(eg_list,nb_shot)
        eg_prompt = """#EXAMPLES: \n""" + "\n".join(eg_prompt_temp)
        
    return eg_prompt

def set_sch():
    sch = pd.DataFrame()
    sch['sch_id'] = ['Example', 'CauseToEffect', 'PracticalReasoning', 'Consequences', 'PopularOpinion', 'Values', 'Analogy', 'Sign', 'FearAppeal', 'DangerAppeal',
                     'VerbalClassification', 'ExpertOpinion', 'Bias', 'Alternatives',  'ArgumentFromAuthority', 'PopularPractice', 'PositiveConsequences', 
                     'CircumstantialAdHominem', 'PositionToKnow', 'GenericAdHominem', 'DirectAdHominem', 'NegativeConsequences', 'SignFromOtherEvents','Ad hominem']
    
    sch['name'] = ['EXAMPLE', 'CAUSE TO EFFECT', 'PRACTICAL REASONING', 'CONSEQUENCES', 'POPULAR OPINION', 'VALUE', 'ANALOGY', 'SIGN', 'FEAR AND DANGER APPEALS',
                     'DANGER APPEAL', 'VERBAL CLASSIFICATION', 'EXPERT OPINION', 'BIAS', 'ALTERNATIVES', 'EXPERT OPINION','POPULAR PRACTICE', 'POSITIVE CONSEQUENCES',
                     'CIRCUMSTANTIAL AD HOMINEM', 'POSITION TO KNOW', 'GENERIC AD HOMINEM', 'DIRECT AD HOMINEM', 'NEGATIVE CONSEQUENCES', 'SIGN','AD HOMINEM'] 
    

                     
    sch['def'] = ['This scheme involves reasoning from a specific case or instance to a general conclusion, suggesting that what holds in the example applies more broadly.',
                  'This scheme reasons that if a certain cause occurs, it will lead to a specific effect, based on a causal relationship.',
                  'This scheme involves an agent reasoning from a goal to an action that is a means to achieve that goal (e.g., "I want G, doing A achieves G, so I should do A").',
                  'This scheme bases a conclusion on the positive or negative outcomes of a proposed action, arguing for or against it based on those consequences.',
                  'This scheme argues that a proposition is true or should be accepted because it is widely believed by the majority.',
                  'This scheme reasons that an action should be taken or avoided because it aligns with or conflicts with an agent’s values (e.g., "V is good, so I should pursue G that promotes V").',
                  'This scheme draws a conclusion about one case by comparing it to a similar case where the conclusion is known to hold.',
                  'This scheme infers a conclusion based on an observable sign or indicator that suggests the presence of a condition or event.',
                  'This scheme urges action or avoidance based on the fear of a harmful outcome if the action isn’t taken or is taken.',
                  'This scheme warns against an action by citing a specific danger it poses, emphasizing risk avoidance.',
                  'This scheme applies a general rule or property to a specific case based on how the case is classified linguistically.',
                  'This scheme concludes that a proposition is true because an expert in the relevant field asserts it.',
                  'This scheme attacks an argument by alleging that the source is biased, thus undermining its credibility.',
                  'This scheme reasons that one option should be chosen (or avoided) by comparing it to other possible options.',
                  'This scheme supports a conclusion based on the assertion of an authoritative figure, often implying expertise or status.',
                  'This scheme justifies an action or belief because it is commonly practiced by many people.',
                  'This scheme argues for an action because it will plausibly lead to good outcomes',
                  'This scheme attacks an opponent’s argument by alleging inconsistency between their actions and their stated position.',
                  'This scheme concludes a proposition is true because the source is in a position to know about it (e.g., firsthand experience).',
                  'This scheme undermines an argument by attacking the character or credibility of the arguer.',
                  'This scheme directly attacks the arguer personally (e.g., calling them a liar) to discredit their argument.',
                  'This scheme argues against an action because it will lead to bad outcomes.',
                  'This scheme infers a conclusion from events or signs that indirectly indicate a condition or outcome.',
                  'This scheme attacks a person’s character, circumstances, or credibility to undermine their argument, rather than directly refuting its content, within a dialogical exchange.']
                  
                  
    sch['cq_template']= ['Is the example representative of the broader category or situation? Are there significant counterexamples that undermine the generalization? Is the example relevant to the conclusion being draw',
                         'Is there sufficient evidence that the cause reliably produces the effect? Could other factors intervene to prevent the effect from occurring? Is the causal link based on correlation rather than proven causation?',
                         'What other goals might conflict with G? Are there alternative actions to A that could also achieve G? Is A the most efficient means to achieve G? Is it practically possible for me to carry out A? What are the potential side effects or consequences of doing A?',
                         'Are the predicted consequences likely to occur if the action is taken? Are there other consequences (positive or negative) that haven’t been considered? Is the evaluation of the consequences as good or bad justified?',
                         'Is the opinion truly held by a significant majority? Does the majority have reliable evidence or expertise to justify their belief? Could the majority be mistaken or influenced by bias?',
                         'Is value V genuinely positive/negative as judged by the agent? Does pursuing V conflict with other values the agent holds? Is the link between the action and the promotion of V well-supported?',
                         'Are the two cases sufficiently similar in relevant respects? Are there significant differences that undermine the analogy? Is the conclusion in the known case well-established?',
                         'Is the sign a reliable indicator of the conclusion? Could the sign be present without the conclusion being true? Are there alternative explanations for the sign?',
                         'Is the feared outcome realistically likely to occur? Is the fear disproportionate to the evidence of danger? Are there other ways to mitigate the feared outcome without the proposed action?',
                         'Is the danger credible and supported by evidence? Could the danger be avoided through means other than the proposed action? Is the danger significant enough to justify the conclusion?',
                         'Is the classification of the case accurate and appropriate? Does the general rule reliably apply to all cases under this classification? Is the classification ambiguous or contested?',
                         'How credible is the expert as a source? Is the expert an authority in the field relevant to the proposition? What exactly did the expert assert? Is the expert personally reliable and trustworthy? Is the expert’s claim consistent with other experts? Is the expert’s assertion backed by evidence?',
                         'Is there clear evidence of bias in the source? Does the alleged bias directly affect the truth of the argument’s conclusion? Could the argument still hold despite the bias?',
                         'Have all relevant alternatives been considered? Are the alternatives fairly evaluated? Is the chosen alternative clearly superior based on the criteria?',
                         'Is the authority credible in the relevant domain? Does the authority’s assertion align with evidence or reason? Is reliance on authority justified over independent evidence?',
                         'Is the practice widespread enough to be considered popular? Does the practice’s popularity indicate its correctness or value? Are there reasons the practice might be flawed despite its popularity?',
                         'Are the positive consequences likely to occur? Are there potential negative consequences that outweigh the positive ones? Is the assessment of the consequences as positive well-founded?',
                         'Is the alleged inconsistency real and relevant to the argument? Does the inconsistency undermine the argument’s validity? Could the argument still hold despite the personal inconsistency?',
                         'Is the source genuinely in a position to know about the proposition? Is the source honest and trustworthy? Did the source actually assert the proposition?',
                         'Is the character flaw relevant to the argument’s validity? Is there evidence supporting the attack on the arguer’s character? Could the argument stand independently of the arguer’s character?',
                         'Does the personal attack relate to the argument’s content? Is the attack substantiated with evidence? Can the argument be evaluated apart from the personal attack?',
                         'Are the negative consequences probable? Are there positive consequences that might offset the negative ones? Is the judgment of the consequences as negative reasonable?',
                         'Do the events reliably point to the conclusion? Could the events be explained by something else? Are the events sufficiently connected to the conclusion?',
                         'Is the attack relevant to the argument’s validity? Is the personal critique supported by evidence? Can the argument stand despite the attack?',]
    
    sch['paper_template'] =    ["Is it actually the case that '<subjectA>' '<featF>' '<featG>'? Is there evidence on this claim? Is '<subjectA>' actually a typical case of other '<subject>' that '<featF>'? How widely applicable is the generalisation? Are there special circumstances pertaining to '<subjectA>' that undermine its generalisability to other '<subject>' that '<featF>'?",
                        "How strong is the generalisation that if '<eventA>' then '<eventB>'? Are there other factors in this particular case that could have interfered with the event of '<eventB>'?",
                        "Are there other relevant goals that conflict with '<goalG>'? Are there alternative actions to '<eventA>' to achieve '<goalG>'? If so, which is the most efficient action? Could '<eventA>' have consequences that we should take into account? Is it practically possible?",
                        "If '<eventA>', will '<eventB>' occur? What evidence supports this claim? How likely are the consequences? What other consequences should also be taken into account if '<neg>' '<eventA>'?",
                        "What evidence supports that '<eventA>' is generally accepted as true? Even if '<eventA>' is generally accepted as true, are there any good reasons for doubting that it is true?",
                        "Is '<valueV>' seen as '<direction>' for most people? Are there reasons to believe that '<valueV>' is not '<direction>' in this situation? Will a subject that sees '<valueV>' as not '<direction>' agree with retaining commitment to '<goalG>'?",
                        "Are '<C1>' and '<C2>' similar in the respect cited? Is '<eventA>' true in '<C1>'? Are there differences between '<C1>' and '<C2>' that would tend to undermine the force of the similarity cited? Is there some other case that is also similar to '<C1>', but in which '<eventA>' is false?",
                        "Is there a proved relation between '<eventB>' and '<eventA>'? Are there any events other than '<eventB>' that would more reliably account for '<eventA>'?",
                        "Is '<eventB>' bad? Why and to whom is it bad? Is '<eventA>' a way to prevent '<eventB>'? Is it practically possible for '<eventA>' to happen? Are there other consequences from '<eventA>'?",
                        "If '<eventA>', will '<eventB>' occur? What evidence supports this claim? Why is '<eventB>' a danger? To whom is '<eventB>' a danger? Is there a way of preventing '<eventA>'? Are there other consequences of preventing '<eventA>' that we should take into account?",
                        "Is it the case that '<subjectA>' '<featF>', or is there room for doubt? Is there a proved relation between situations in which '<featF>' and situations in which '<featG>'? Is it possible for the particular case of '<subjectA>' that '<featG>' is not the case?",
                        "Is '<expertE>' a genuine expert in '<domainD>'? Did '<expertE>' really assert that '<eventA>'? Is '<expertE>'s pronouncement directly quoted? If not, is a reference to the original source given? Can it be checked? If '<expertE>'s advice is not quoted, does it look like important information or qualifications may have been left out? Is what '<expertE>' said clear? Are there technical terms used that are not explained clearly? Is '<eventA>' relevant to domain '<domainD>'? Is '<eventA>' consistent with what other experts in '<domainD>' say? Is '<eventA>' consistent with known evidence in '<domainD>'?",
                        "What evidence is there that '<subjectA>' is '<subjectB>'? Could '<subjectA>' have taken evidence on many sides even if '<subjectA>' is '<subjectB>'? Does the matter of '<event>' require '<subjectA>' to take evidence on many sides?",
                        "Can '<eventB>' happen even if '<eventA>' is the case? Is '<eventA>' plausibly not the case? What evidence supports this claim?",
                        "Is '<expertE>' a genuine expert in '<domainD>'? Did '<expertE>' really assert that '<eventA>'? Is '<expertE>'s pronouncement directly quoted? If not, is a reference to the original source given? Can it be checked? If '<expertE>'s advice is not quoted, does it look like important information or qualifications may have been left out? Is what '<expertE>' said clear? Are there technical terms used that are not explained clearly? Is '<eventA>' relevant to domain '<domainD>'? Is '<eventA>' consistent with what other experts in '<domainD>' say? Is '<eventA>' consistent with known evidence in '<domainD>'?",
                        "What actions or other indications show that '<large_majority>' accept that '<eventA>' is the right thing to do? Even if '<large_majority>' accepts '<eventA>' is the right thing to do, are there grounds for thinking they are justified in accepting it as a prudent course of action?",
                        "If '<eventA>', will '<eventB>' occur? What evidence supports this claim? How likely are the consequences? What other consequences should also be taken into account if '<neg>' '<eventA>'?",
                        "Does '<argument1>' imply '<eventA>'? Can the practical inconsistency between '<subjectA>'s commitments and '<eventA>' be identified? Can it be shown by evidence? Could it be explained by further dialogue? Does the inconsistency between '<subjectA>'s commitments and '<eventA>' result in a decrease of credibility for '<subjectA>'? Does '<subjectA>'s argument depend on its credibility in this context?",
                        "Is '<subjectA>' in a position to know whether '<eventA>'? Is '<subjectA>' an honest (trustworthy, reliable) source? Did '<subjectA>' assert that '<eventA>'?",
                        "How does the allegation made affect the reliability of '<subjectA>'? Is the reliability of '<subjectA>' relevant in the current dialogue?",
                        "Does '<argument1>' imply '<eventA>'? Can the practical inconsistency between '<subjectA>'s commitments and '<eventA>' be identified? Can it be shown by evidence? Could it be explained by further dialogue? Does the inconsistency between '<subjectA>'s commitments and '<eventA>' result in a decrease of credibility for '<subjectA>'? Does '<subjectA>'s argument depend on its credibility in this context?",
                        "If '<eventA>', will '<eventB>' occur? What evidence supports this claim? How likely are the consequences? What other consequences should also be taken into account if '<neg>' '<eventA>'?",
                        "Is there a proved relation between '<eventB>' and '<eventA>'? Are there any events other than '<eventB>' that would more reliably account for '<eventA>'?",
                        "Does '<argument1>' imply '<eventA>'? Can the practical inconsistency between '<subjectA>'s commitments and '<eventA>' be identified? Can it be shown by evidence? Could it be explained by further dialogue? Does the inconsistency between '<subjectA>'s commitments and '<eventA>' result in a decrease of credibility for '<subjectA>'? Does '<subjectA>'s argument depend on its credibility in this context?",]

    sch['papercq_template'] =sch['cq_template'] + "\n\n" + sch['paper_template']

    sch.to_csv('./DATA/proc/sq/sch_def.csv', index=False)                       


def prompt_question_sch(essay_segmented,nb_questions,sch_list =None, nb_shot=0, eg_list=[],prompt_essay = None,log=None,idx_essay=None, idx_repetition=None,task ='H'):
    eg_prompt = prompt_compo_eg(nb_shot, eg_list)
    sch_def = pd.read_csv('./DATA/proc/sq/sch_def.csv')
    if 'H' in  task and 'E' in  task:
        template_type = 'papercq_template'
    elif 'H' in  task:
        template_type = 'cq_template'
    elif 'E' in  task:
        template_type = 'paper_template'
        
    sch_prompt = ''
    for i_sch in range(len(sch_list)):
        sch_raw = sch_def.loc[sch_def['sch_id'] == sch_list[i_sch]]
        sch_prompt += f"{sch_raw['name'].values[0]}: {sch_raw['def'].values[0]} TEMPLATES OF CRITICAL QUESTIONS: {sch_raw[f'{template_type}'].values[0]}"
    prompt_query =f""" You are a very strict critical and sceptical judge.
    # ARGUMENTATION SCHEMES: Argumentation schemes are stereotypical patterns of inference that capture common types of defeasible arguments, i.e. arguments that are plausible but open to rebuttal. Each scheme represents a form of reasoning with typical premises and a conclusion. Here are the argumentation schemes present in the essay along with their definitions and template of critical questions:
    {sch_prompt}
    #QUERY: 
    Use the provided scheme and their template of critical questions to generate {nb_questions} critical questions to evaluate the arguments in the given essay. Critical questions are inquiries designed to assess the strength or fallibility of an argument by revealing assumptions in its premises.
    List one question per line. Keep questions simple and direct, without explaining their relevance""" 
    
    # prompt_query =f""" You are a very strict critical and sceptical judge.
    # # ARGUMENTATION SCHEMES: Argumentation schemes are stereotypical patterns of inference that capture common types of defeasible arguments, i.e. arguments that are plausible but open to rebuttal. Each scheme represents a form of reasoning with typical premises and a conclusion. Here is the argumentation scheme present in the essay along with its definition and template of critical questions:
    # {sch_prompt}
    # #QUERY: 
    # Use the provided scheme and its template of critical questions to generate {nb_questions} critical questions to evaluate the arguments in the given essay. Critical questions are inquiries designed to assess the strength or fallibility of an argument by revealing assumptions in its premises.
    # List one question per line. Keep questions simple and direct, without explaining their relevance""" 

    prompt = eg_prompt +\
        """\n#ESSAY:\n""" + essay_segmented + """\n""" + prompt_query #+ outputrequirement
        
    #remove extra spaces
    prompt.replace('\n', ' ') 
    prompt= ' '.join(prompt.split())

    if idx_essay ==0 and idx_repetition==0 and log is not None:
        log('\nPrompt quest:\n'+prompt)
        
    return prompt 


def prompt_question(essay_segmented,nb_questions, nb_shot=0, eg_list=[],prompt_essay = None,log=None,idx_essay=None, idx_repetition=None):
    eg_prompt = prompt_compo_eg(nb_shot, eg_list)
    prompt_query =f"""

#QUERY:
    You are a very strict critical and sceptical judge. List {nb_questions} critical questions that should be raised before accepting the arguments in the given essay. Critical questions are the set of enquiries that should be asked in order to judge if an argument is good or fallacious by unmasking the assumptions held by the premises of the argument.
            Give one question per line. Make the questions simple, and do not give any explanation regarding why the question is relevant.""" 

    prompt = eg_prompt +\
        """\n#ESSAY:\n""" + essay_segmented + """\n""" + prompt_query #+ outputrequirement
        
    #remove extra spaces
    prompt.replace('\n', ' ') 
    prompt= ' '.join(prompt.split())

    if idx_essay ==0 and idx_repetition==0 and log is not None:
        log('\nPrompt quest:\n'+prompt)
        
    return prompt 

def prompt_select_question(essay_segmented,questions,select_best = True, nb_shot=0, eg_list=[],task = None,prompt_essay = None,log=None,idx_essay=None, idx_repetition=None,sch_list =None):
        
    #set few shot
    eg_prompt = prompt_compo_eg(nb_shot, eg_list)

    sch_def = pd.read_csv('./DATA/proc/sq/sch_def.csv')
    if 'H' in  task and 'E' in  task:
        template_type = 'papercq_template'
    elif 'H' in  task:
        template_type = 'cq_template'
    elif 'E' in  task:
        template_type = 'paper_template'
        
    if task =='G_C':
        # set query: ask for 3 best, or 3 random/ less good
        prompt_query =f""" You are a critical judge.  Critical questions are the set of enquiries that should be asked in order to judge if an argument is good or fallacious by unmasking the assumptions held by the premises of the argument.
        Select the 3 best critical questions among this list that should be raised before accepting the arguments in the essay. Give one question per line, and do not give any explanation regarding why the question is relevant."""
        prompt = eg_prompt +\
            """\n#ESSAY:\n""" + essay_segmented +"""\n #QUESTIONS\n"""+ questions + """ \n#QUERY: \n""" + prompt_query #+ outputrequirement 
    else:    
        sch_prompt = ''
        for i_sch in range(len(sch_list)):
            sch_raw = sch_def.loc[sch_def['sch_id'] == sch_list[i_sch]]
            sch_prompt += f"{sch_raw['name'].values[0]}: {sch_raw['def'].values[0]} TEMPLATES OF CRITICAL QUESTIONS: {sch_raw[f'{template_type}'].values[0]}"
    
        prompt_query =f""" You are a critical judge. #ARGUMENTATION SCHEMES: Argumentation schemes are stereotypical patterns of inference that capture common types of defeasible arguments, i.e. arguments that are plausible but open to rebuttal. Each scheme represents a form of reasoning with typical premises and a conclusion. Here are the argumentation schemes present in the essay along with their definitions and template of critical questions:
        {sch_prompt}  Critical questions are the set of enquiries that should be asked in order to judge if an argument is good or fallacious by unmasking the assumptions held by the premises of the argument.
        """
        prompt =  prompt_query +"""\n#ESSAY:\n""" + essay_segmented +"""\n #QUESTIONS\n"""+ questions +   """#QUERY: Select the 3 best critical questions among this list that should be raised before accepting the arguments in the essay.
        If some questions are redundant, theses questions must be important: select the most relevant one.
        Give one question per line, and do not give any explanation regarding why the question is relevant. """#+ outputrequirement

    
        # prompt="""\n#ESSAY:\n""" + essay_segmented +"""\n #QUESTIONS\n"""+ questions +   """#QUERY: Select the 3 best critical questions among this list that should be raised before accepting the arguments in the essay.
        #     If some questions are redundant, theses questions must be important: select the most relevant one.
        #     Give one question per line, and do not give any explanation regarding why the question is relevant."""
        #remove extra spaces
        prompt.replace('\n', ' ') 
        prompt= ' '.join(prompt.split())

        if idx_essay ==0 and idx_repetition==0 and log is not None:
            log('\n selec quest:\n'+prompt)
            
        return prompt 
    

