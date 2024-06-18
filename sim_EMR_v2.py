import sys
import os
sys.path.insert(0, os.path.abspath('../cycleSim'))
from realSim import simulator_base,downsample,get_mSF
from trialSimulator import RR50, MPC, calculate_fisher_exact_p_value, calculate_MPC_p_value
import gc   
#import ollama
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import seaborn as sns
#import json
import pandas as pd
#from pydantic import BaseModel,ValidationError
#from typing import List

from langchain_community.llms import Ollama
import time


def build_trial_set(patients_per_arm = 112,placebo_efficacy = 0, drug_efficacy = 0.39, baseline_dur = 2, test_dur = 3,
                    fname_raw = 'raw',fname_sum = 'sum',true_sum='true',IDstart=0,noSum=True,rawTF=True,
                    LLM_gen_name='llama2:13b'):
    
    min_sz_rate = 4  # minimum seizure rate per month to be included in the trial
    
    # make a 2 by patients_per_arm array of patient IDs which are shuffled randomly
    #IDlist = np.reshape(np.random.permutation(np.arange(patients_per_arm*2)),(2,patients_per_arm))
    IDlist = np.reshape(IDstart + np.arange(patients_per_arm*2),(2,patients_per_arm))

    # this is the possible symptom list and probability list for each symptom
    # these numbers come from Krauss et al 2022
    list_of_sx = ['dizziness', 'somnolence', 'fatigue', 'nausea', 'constipation', 'balance disorder',
                'nystagmus', 'ataxia', 'dysarthria', 'vomiting', 'back pain', 'vertigo', 
                'fall', 'upper respiratory infection', 'headaches','diplopia','gait disturbance','decreased appetite']
    # sx_probs[0,:] is the probability of a symptom for the placebo group
    # sx_probs[1,:] is the probability of a symptom for the drug group
    sx_probs =  [   [0.14, 0.08, 0.06, 0.01, 0.01, 0.03,
                    0.01, 0.01, 0,    0,    0.03, 0.03,
                    0.06, 0.09, 0.06, 0.02, 0.03, 0.01],
                    [0.33, 0.37, 0.24, 0.09, 0.09, 0.09,
                    0.06, 0.06, 0.06, 0.05,  0.05, 0.05,
                    0.06, 0.03, 0.11, 0.15, 0.08, 0.05]]
    
    drugNames = ['X13737','X93461']

    temp_gen = 1.0
    temp_sum = 0.0
    #LLM_gen_name = 'llama2:13b'      
    LLM_sum_name = 'mistral'


    for arm in range(2):
        with open(f'{fname_raw}_arm{arm}.txt', 'a') as f_raw, open(f'{fname_sum}_arm{arm}.txt', 'a') as f_sum, open(f'{true_sum}_arm{arm}.tsv', 'a') as f_true:
        #with open(f'{fname_raw}_arm{arm}_part{P}.txt', 'a') as f_raw, open(f'{fname_sum}_arm{arm}_part{P}.txt', 'a') as #f_sum, open(f'{true_sum}_arm{arm}_part{P}.tsv', 'a') as f_true:
    
            if arm==0:
                efficacy_value = placebo_efficacy
            else:
                efficacy_value = drug_efficacy
            
            for i in range(patients_per_arm):
                print(f'\nArm {arm}, patient {IDlist[arm,i]}',end='')
                idnum = IDlist[arm,i]
                name = f'Patient ID {idnum}'
                drug_name = drugNames[arm]
                
                # generate number of seizures in baseline and test periods
                ####
                mSF = 0
                while mSF<min_sz_rate:
                    mSF = get_mSF(requested_msf=-1)    # choose a monthly seizure frequency

                temp = simulator_base(sampRATE=1,number_of_days=baseline_dur*30,defaultSeizureFreq=mSF)
                month_diary = downsample(temp,30)
                baseline_sz = np.sum(month_diary).astype(int)

                mSF = mSF * (1-efficacy_value)      # modulate seizure frequency by efficacy
                temp = simulator_base(sampRATE=1,number_of_days=test_dur*30,defaultSeizureFreq=mSF)
                month_diary = downsample(temp,30)
                test_sz = np.sum(month_diary).astype(int)
                ##
                
                selected_sx = []
                for s_count,symptom in enumerate(list_of_sx):
                    if np.random.rand()<sx_probs[arm][s_count]:
                        selected_sx.append(symptom)
                # if there are any symptoms, add commas to the list
                if len(selected_sx)>0:
                    selected_sx_str = ', '.join(selected_sx)               
                else:   # no symptoms at all
                    selected_sx_str = 'no symptoms'
                
                if rawTF==True:
                    R1,R2 = make_one_patient(name = name, idnum= idnum, drug_name=drug_name , baseline_sz=baseline_sz, 
                                test_sz=test_sz, baseline_dur=baseline_dur, 
                                test_dur=test_dur,selected_sx_str = selected_sx_str,noSum=noSum,
                                LLM_gen_name=LLM_gen_name,LLM_sum_name=LLM_sum_name,arm=arm)
                    f_raw.write(R1[0] + R1[1])
                    if noSum==False:
                        f_sum.write(R2[0] + R2[1])
                f_true.write(f'{IDlist[arm,i]}\t{arm}\t{baseline_sz}\t{test_sz}\t{selected_sx_str}\n')
    #del LLM_gen
    #del LLM_sum
    #gc.collect()
    print('\nDone.')

def getLLM_response(thePrompt,LLM_name,temp=0.7,time_out=120,streamprint=False):
    # given inputs, return the response from the LLM up until the time_out seconds
    llm = Ollama(model=LLM_name,temperature=temp)

    response = llm.stream(thePrompt)
    t  = time.time()
    K = ''
    for part in response:
        K += part
        if time.time()-t>time_out:
            print('Time out!',end='')
            sys.exit()
            break
        if streamprint==True:
            print(part,end='')

    print('...',end='')

    del llm
    gc.collect()
    return K

def getLLM_response_withLLM(thePrompt,llm,time_out=120):
    # given inputs, return the response from the LLM up until the time_out seconds
    
    response = llm.stream(thePrompt)
    t  = time.time()
    K = ''
    for part in response:
        K += part
        if time.time()-t>time_out:
            print('Time out!',end='')
            sys.exit()
            break
    print('...',end='')

    gc.collect()
    return K

def make_one_patient(name = 'Patient ID 999',idnum=999,drug_name='ID3838' , baseline_sz=8, test_sz=8, baseline_dur=2, 
                    test_dur=3, selected_sx_str='no symptoms',noSum=False,LLM_gen_name='llama2:13b',LLM_sum_name = 'mistral',arm=0):

    gc.collect() # clean up the memory
    time_out = 120  # max seconds for generating one response. This is a safety feature to prevent too much output.
    temp_gen = 1.0
    temp_sum = 0.0
    
    # Create a client object
    LLM_gen = Ollama(model=LLM_gen_name,temperature=temp_gen)
    LLM_sum = Ollama(model=LLM_sum_name,temperature=temp_sum)


    style_list = [
                'Write in the style of a neurologist clinic note. Include a complete neurologic exam and a brief assesssment and plan.',
                'Write in the style of a narrative form, with complete sentences and paragraphs only. Do noy use any bullet points. The entire response should be 2-4 paragraphs long.',
                'write in a minimalist style, with very terse language and only minimal detail. Include identifying information, HPI, brief neurologic examination with pertinent positives only, and a brief assessment and plan. Do not use complete sentences - instead use bullet points.',
                'write in the expansive style of a very erudite academic professor of neurology writing a clinic note with many exrtraneous details, including some of the words from the patient, things about the patient demeanor, medication adherence, a review of laboratory findings and imaging findings, and a long assessment that includes various considerations. Include sections for Identifcation, HPI, Subjective, Medications, Allergies, Social history, Review of systems, General Physical examination, Comprehensive neurologic exam, Labs, Imaging, Assessment, Plan, and a signature.']
    style_choice = np.random.choice(style_list,size=2,replace=True)

    age = np.random.randint(18,100)
    sex = np.random.choice(['male','female'])

    list_of_dx = ['dermatitis','diabates','hypertension','arthritis','hypothyroidism',
                'seasonal allergies','asthma','depression','anxiety',
                'bipolar disorder','schizophrenia','ADHD','autism','alcoholism']
    # Randomly choose a number between 0 and 3
    num_dx = np.random.randint(0, 4)
    # Randomly select the words
    selected_words = np.random.choice(list_of_dx, num_dx, replace=False)
    selected_words_str = ', '.join(selected_words)
    list_of_epilepsy = ['temporal lobe epilepsy','frontal lobe epilepsy',
                        'parietal lobe epilepsy','occipital lobe epilepsy',
                        'generalized epilepsy','multifocal epilepsy','epilepsy']
    epilepsy_type = np.random.choice(list_of_epilepsy)

    # Specify the date
    # Specify the start and end dates
    study_start_enrollment = datetime.strptime("2021-01-01", "%Y-%m-%d")
    study_end_enrollment = datetime.strptime("2022-12-31", "%Y-%m-%d")
    # Calculate the difference between the two dates
    time_between_dates = study_end_enrollment - study_start_enrollment
    days_between_dates = time_between_dates.days
    # Generate a random number of days to add to the start date
    random_number_of_days = np.random.randint(days_between_dates)
    # Add the random number of days to the start date
    study_enrollment_obj = study_start_enrollment + timedelta(days=random_number_of_days)
    
    # Add months to the date
    new_date1_obj = study_enrollment_obj + relativedelta(months=baseline_dur)
    # Convert the new date object back to a string
    new_date1_str = new_date1_obj.strftime("%Y-%m-%d")
    new_date2_obj = study_enrollment_obj + relativedelta(months=test_dur)
    # Convert the new date object back to a string
    new_date2_str = new_date2_obj.strftime("%Y-%m-%d")


    mlist= [
        f'{style_choice[0]} Write a clinic note visit #1 dated {new_date1_str} for {name}, a {age} year old {sex} with a history of {selected_words_str} and {epilepsy_type}. The patient has had {baseline_sz} seizures in the past {baseline_dur} months. The patient is in a randomized controlled trial. No change in medications will be made other than adding the investigational drug for epilepsy that will be taken for {test_dur} months. In the note, include a complete neurological examination.',
        f'{style_choice[1]} Write a clinic note vist #2 dated {new_date2_str} for {name}, a {age} year old {sex} with a history of {selected_words_str} and {epilepsy_type}. The patient is in a randomized controlled trial. The patient took the investigational drug ID {drug_name} for epilepsy over the past {test_dur} months and in that time has had {test_sz} seizures. The patient reported {selected_sx_str}. The investigational drug will be discontinued now that the study is complete. In the note, include a complete neurological examination.',
        ]
    R1 = ['','']
    R2 = ['','']

    
    for m, msg in enumerate(mlist):
        # Get the response from the model
        #K1 = getLLM_response(msg,LLM_gen_name,temp=temp_gen,time_out=time_out) 
        K1 = getLLM_response_withLLM(msg,LLM_gen,time_out=time_out)
        R1[m] = f'-------------- PATIENT ID {idnum} -- VISIT {m+1} -----------\n' + \
            K1 + \
            '\n----------------END OF PATIENT ENCOUNTER---------------\n'
        

        if noSum==False:
            m2 = f'Please summarize the following clinic note. Specifically, produce a 2 line output. The output should be include only the following info (one element per lime): num_seizures - number of seizures since last visit and symptoms - a list of symptoms reported by the patient. {R1[m]}'
            #encounter_sum2 = getLLM_response(m2,LLM_sum_name,temp=temp_sum,time_out=time_out)
            encounter_sum2 = getLLM_response_withLLM(m2,LLM_sum,time_out=time_out)
            R2[m] = f'ID - {idnum}\nVisit - {m+1}\nArm - {arm}\n' + encounter_sum2 + \
                '\n----------------END OF PATIENT ENCOUNTER---------------\n'
        else:
            R2[m] = ''

    del LLM_gen
    del LLM_sum
    gc.collect()
    
    return R1,R2


def load_true_and_analyze(tsv_file='trueB_summary_all.tsv',headerTF=False):
    # Load the data into a DataFrame
    if headerTF==False:
        df = pd.read_csv(tsv_file, delimiter='\t', names=['ID','arm', 'baseline_sz', 'test_sz', 'sx'])
    else:
        df = pd.read_csv(tsv_file, delimiter='\t')
    
    # error checking on numbers
    df['valid'] = np.where((pd.to_numeric(df['baseline_sz'], errors='coerce').notnull()) & (pd.to_numeric(df['test_sz'], errors='coerce').notnull()), 1, 0)
    dfNum = df[df['valid'] == 1].copy()

    dfNum['baseline_sz'] = dfNum['baseline_sz'].astype(np.int64)
    dfNum['test_sz'] = dfNum['test_sz'].astype(np.int64)

    # Calculate the 'PC' column
    base_rate = dfNum['baseline_sz']  / 2
    test_rate = dfNum['test_sz']/ 3
    df['PC'] = 100 * (base_rate- test_rate) / base_rate
    df['valid'] = np.logical_and(np.logical_not(np.isinf(df['PC'])),np.logical_not(np.isnan(df['PC'])))
    dfv = df[df['valid'] == 1].copy()
    d_drug = dfv[dfv['arm'] == 1].PC
    d_placebo = dfv[dfv['arm'] == 0].PC

    # Display the DataFrame
    #print(df)
    print(f'Filenae={tsv_file}...')
    x = [len(dfv), RR50(d_placebo), RR50(d_drug), calculate_fisher_exact_p_value(d_placebo,d_drug), \
        MPC(d_placebo), MPC(d_drug), calculate_MPC_p_value(d_placebo,d_drug)]
    print(f'Number of valid patients = {x[0]}')
    print(f'RR50 PCB = {np.round(x[1])}, RR50 DRG = {np.round(x[2])} RR50p = {x[3]}')
    print(f'MPC PCB = {np.round(x[4])}, MPC DRG = {np.round(x[5])}, MPCp = {x[6]}')
    return x

# Function to replace symptom names
def replace_symptoms(symptom_list):
    # Define replacements here; could be expanded for other symptoms
    replacements = {
        'upper respiratory infection': 'uri',
        'headaches': 'headache',
        'no symptoms': 'none',
        'none reported' : 'none',
        'backpain' : 'back pain',
        'mild weakness in left arm and leg' : 'weakness',
        'weakness in left arm and leg' : 'weakness',
        '':'none',
        ' ':'none',
        '\t':'none'
        # Add more replacements as needed
    }
    # Replace each symptom in the list if it's in the replacements dict
    return [replacements.get(symptom, symptom) for symptom in symptom_list]


def summarize_sx(tsv_file='trueX1_summary_all.tsv',out_df='sx_sum.tsv',headerTF=False,do_all=False):
    if headerTF==False:
        df = pd.read_csv(tsv_file, delimiter='\t', names=['ID','arm', 'baseline_sz', 'test_sz', 'sx'])
    else:
        df = pd.read_csv(tsv_file, delimiter='\t')

    
    df['sx'] = df['sx'].str.lower()
    df['sx'] = df['sx'].fillna('none')
    df['sx_list'] = df['sx'].str.split(', ')
    df['sx_list'] = df['sx_list'].fillna('none')
    N = df.shape[0]

    # apply some replacement rules to make  the visualization nicer
    df['sx_list'] = df['sx_list'].apply(replace_symptoms)    

    sx_frequency_0 = df[df['arm'] == 0]['sx_list'].explode().value_counts()
    sx_frequency_1 = df[df['arm'] == 1]['sx_list'].explode().value_counts()

    df_sx = pd.DataFrame({'arm0':np.round(100*sx_frequency_0/N),
                        'arm1':np.round(100*sx_frequency_1/N)})
    df_sx = df_sx.fillna(0)
    if do_all==True:
        # here I am showing you eveyerthing
        nonZero = np.logical_or(df_sx['arm0']>0,df_sx['arm1']>0)
    else:
        # here I am being brief and showing only things that happen in both arms
        nonZero = np.logical_and(df_sx['arm0']>0,df_sx['arm1']>0)

    df_sx[nonZero].to_csv(out_df,sep='\t',index=True)
    return df_sx[nonZero].copy()

def build_summaries(raw_name,summary_name):

    LLM_sum = 'mistral'
    temp_sum = 0.0  # temperature of LLM for summarization
    time_out = 120  # max seconds for generating one response. This is a safety feature to prevent too much output.
    
    # Define the delimiter that marks the end of a patient encounter
    delimiter = "----------------END OF PATIENT ENCOUNTER---------------"

    # Read the entire file content
    with open(raw_name, 'r') as file_in:
        content = file_in.read()

    # Split the content into individual encounters based on the delimiter
    encounters = content.split(delimiter)

    with open(summary_name, 'w') as file_out:
        # Iterate through each encounter, except the last empty string if present
        # The last split is likely to be an empty string since the file ends with the delimiter
        for encounter in encounters[:-1]: 
            #print(encounter.strip() + "\n" + delimiter + "\n")
            encounter = encounter.strip() 
            lines = encounter.split('\n')
            line1 = lines[0]
            all_else = '\n'.join(lines[1:])
            m1 = f'Please summarize the following clinic note. Specifically, produce a 2 line output. The output should be include only the following info (one element per lime): ID - the patient ID number, visit - vist number 0 or visit number 1. {line1}'
            encounter_sum1 = getLLM_response(m1,LLM_sum,temp=temp_sum,time_out=time_out)
            #print(encounter)
            m2 = f'Please summarize the following clinic note. Specifically, produce a 2 line output. The output should be include only the following info (one element per lime): num_seizures - number of seizures since last visit and symptoms - a list of symptoms reported by the patient. {all_else}'
            encounter_sum2 = getLLM_response(m2,LLM_sum,temp=temp_sum,time_out=time_out)
            print(encounter_sum1 + "\n" + encounter_sum2 + "\n" + delimiter)

            file_out.write(encounter_sum1 + "\n" + encounter_sum2 + "\n" + delimiter + "\n")

def build_my_complete_table(arm0_file, arm1_file,output_prefix='complete_table'):
    # this thing did not work at all really
    LLM_name = 'yarn-mistral:7b-128k'
    temp = 0.0  # temperature of LLM for summarization
    time_out = 1200  # max seconds for generating one response. This is a safety feature to prevent too much output.

    arm_content = ['','']
    with open(arm0_file, 'r') as file:
        arm_content[0] = file.read()
    with open(arm1_file, 'r') as file:
        arm_content[1] = file.read()
    
    for this_arm in range(2):
        thePrompt = f'Here are a collection of clinic visits from many patients. Each patient was seen twice, Visit 1 was after 2 months, and Visit 2 was after 3 additional months. Each visit includes a number of seizures documented. Please generate a data table with the following columns: patient ID = ID number of the patient, visit_1_sz= number of seizures leading up to visit 0, visit_2_sz = number of seizures leading up to Visit 1, symptoms= any symptoms reported in visit 1 that were different from symptoms reported in visit 0. {arm_content[this_arm]}'

        this_arm = getLLM_response(thePrompt,LLM_name,temp=temp,time_out=time_out,streamprint=True)
        output_file = f'{output_prefix}_arm{this_arm}.txt'
        with open(output_file, 'w') as file_out:
            file_out.write(arm_content[this_arm])


def build_result_fig1():
    T = load_true_and_analyze(tsv_file='RCT1_true.txt',headerTF=False)
    E = load_true_and_analyze(tsv_file='Claude2-summary.txt',headerTF=True)
    H = load_true_and_analyze(tsv_file='trial-MD-edited.tsv',headerTF=False)

    # Data

    df = pd.DataFrame({'Source':['True','True','AI','AI','Human','Human'],
                'Arm':['Placebo','Drug','Placebo','Drug','Placebo','Drug'],
                'RR50':[T[1],T[2],E[1],E[2],H[1],H[2]],
                'MPC':[T[4],T[5],E[4],E[5],H[4],H[5]]})


    palette = {'True': 'black', 'AI': 'blue', 'Human': 'red'}

    plt.subplot(1,2,1)
    sns.barplot(data=df, x='Arm', y='RR50', hue='Source',palette=palette)
    plt.title('50% Responder Rate')
    plt.legend().set_visible(False)  # Hide the legend for ax2
    plt.ylabel('Percentage')
    plt.ylim(0,100)
    plt.xlabel('')
    plt.grid(True)
    plt.subplot(1,2,2)
    sns.barplot(data=df, x='Arm', y='MPC', hue='Source',palette=palette)
    plt.title('Median Percent Change')
    plt.ylabel('')
    plt.xlabel('')

    plt.ylim(-50,100)
    plt.grid(True)
    # Show the plot
    plt.savefig('result_fig3.png',dpi=300)
    plt.show()


def build_result_fig2(do_all = True):

    T = summarize_sx(tsv_file='RCT1_true.txt',out_df=f'true_sx-all{do_all}.txt',headerTF=False,do_all=do_all)
    E = summarize_sx(tsv_file='Claude2-summary.txt',out_df=f'claude_sx-all{do_all}.txt',headerTF=True,do_all=do_all)
    H = summarize_sx(tsv_file='trial-MD-edited.tsv',out_df=f'MD_sx-all{do_all}.txt',headerTF=False,do_all=do_all)


    union_index = E.index.union(T.index)
    union_index = H.index.union(union_index)

    TEH = pd.concat([T.reindex(union_index),E.reindex(union_index),H.reindex(union_index)],axis=1)

    # if NA then fill with 0
    TEH = TEH.fillna(0)

    TEH.columns = ['True PCB','True DRG','AI PCB','AI DRG','Human PCB','Human DRG']
    TEH.to_csv(f'Full_sx_table-all{do_all}.csv',index=True)

    pd.set_option('display.width', 1000)
    print(TEH)
    #E = E.reindex(union_index)
    #T = T.reindex(union_index)
    #H = H.reindex(union_index)


    A = pd.Series(TEH['True PCB'],name='True')
    B = pd.Series(TEH['AI PCB'],name='AI')
    C = pd.Series(TEH['Human PCB'],name='Human')
    D = pd.Series(TEH['True DRG'],name='True')
    E = pd.Series(TEH['AI DRG'],name='AI')
    F = pd.Series(TEH['Human DRG'],name='Human')


    df1 = pd.concat([A,B,C],axis=1)
    df2 = pd.concat([D,E,F],axis=1)

    # Sort by 'True DRG' while preserving existing index
    sorted_df2 = df2.sort_values(by='True', ignore_index=False)
    sorted_df1 = df1.loc[sorted_df2.index]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

    # Plot data on each subplot


    ax1 = sorted_df1.plot.bar(ax=ax1, rot=70, color=['black','blue', 'red'])
    ax2 = sorted_df2.plot.bar(ax=ax2, rot=70, color=['black', 'blue', 'red'])

    if do_all==True:
        ax2.legend(loc='upper left')  # Place the legend to the right of the subplot
        ax1.legend().set_visible(False)  # Hide the legend for ax2
    else:        
        ax1.legend(loc='upper right')  # Place the legend to the right of the subplot
        ax2.legend().set_visible(False)  # Hide the legend for ax2
    # Hide x-labels on ax1
    ax1.set_xticklabels([])

    # Add titles and labels
    ax1.set_title('Placebo arm symptoms')
    ax1.set_ylabel('Percentage')
    ax2.set_title('Drug arm symptoms')
    ax2.set_ylabel('Percentage')

    # Rotate x-axis labels for better readability
    if do_all == True:
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=80)

    # Adjust layout
    plt.tight_layout()
    ax1.grid(True)
    ax2.grid(True)
    if do_all==False:
        plt.savefig('result_fig4.png',dpi=300)
    else:
        plt.savefig('result_fig4_all.png',dpi=300)
    plt.show()
