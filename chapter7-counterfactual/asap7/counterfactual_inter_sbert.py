from features.essay_features import extract_essay_features
import os

import xgboost

import joblib
import re
import json
import numpy as np

from openai import OpenAI

from itertools import combinations
import nltk
from nltk.tokenize import sent_tokenize

openai_key = 'PUT_YOUR_OWN_KEY'

prompt = """More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.
"""


def get_essay_score(essay):

    feat = extract_essay_features(essay, prompt)

    script_dir = os.path.dirname(__file__)

    feat_names = 'feature_names_778'

    feature_names_path = os.path.join(
        script_dir, 'features/', feat_names)

    feature_names = joblib.load(feature_names_path)

    model_names = 'model_asap1_778'

    model_names_path = os.path.join(
        script_dir, 'features/', model_names)

    model = joblib.load(model_names_path)

    dmat = xgboost.DMatrix(feat, feature_names=feature_names)
    
    score = model.predict(dmat)[0]

    score_float = np.round(score,3)

    score_int = int(np.round(score))
    
    return score_float, score_int


# Function to apply corrections to the essay
def apply_corrections_grammar(essay, corrections):
    # Tokenize the essay into sentences using NLTK
    sentences = nltk.sent_tokenize(essay)

    # Create a list to store corrected sentences
    corrected_sentences = []

    # Iterate through sentences
    for i, sentence in enumerate(sentences):
        # Apply corrections to the current sentence
        for correction in corrections:
            # Adjust the index by subtracting 1 to match the enumeration starting from 0
            if correction['sentence'] == i + 1:
                # Replace all occurrences of the incorrect text with the correction
                #sentence = sentence.replace(correction['original'], correction['correction'])
                 # Use regular expression with word boundaries to replace only whole words
                sentence = re.sub(r'\b' + re.escape(correction['original']) + r'\b', correction['correction'], sentence)
        # Add the corrected sentence to the list
        corrected_sentences.append(sentence)

    # Join the corrected sentences to form the final essay
    corrected_essay = ' '.join(corrected_sentences)

    return corrected_essay

def apply_corrections_longer(essay, corrections):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(essay)
    
    for correction in corrections:
        corrected_sentence = correction["correction"]
        sentence_index = correction["sentence"]
        
        new_sentence = corrected_sentence
        
        # Replace the original sentence with the corrected one
        sentences[sentence_index - 1] = new_sentence
        
        # Join the sentences back into a single string
    text = ' '.join(sentences)

    return text

def get_all_correction_score(essay, corrections):
    new_essay = apply_corrections_grammar(essay, corrections)
    score = get_essay_score(new_essay)
    return new_essay, score

# Function to find the best combination of changes that improves the essay score
def find_best_combination(essay, corrections, original_score, task="grammar", depth=5):


    #yield f"data: Finding best combination . . .\n\n"

    print("Finding best combination . . .\n")

    # List to store the best combination of corrections that improve the score
    best_combination = []
    best_score = original_score

    # List to store the remaining corrections to try
    remaining_corrections = corrections.copy()

    # While there are corrections left to try
    while remaining_corrections:
        # Variable to store the best correction in this iteration
        best_correction = None
        best_correction_score = best_score
         
        if len(best_combination) + 1 == depth:
            break
        
        print("\n--------------------------------------------------------------")
       
        if len(best_combination) + 1 == 1:
            print(f"Applying {len(best_combination) + 1} correction")
            #yield f"data:Applying {len(best_combination) + 1} correction\n\n"
        else:
            print(f"Applying {len(best_combination) + 1} corrections")
            #yield f"data:Applying {len(best_combination) + 1} corrections\n\n"
     
        print("--------------------------------------------------------------")
        
        # Calculate the score improvement for each correction
        score_improvements = []
        
        # Try each remaining correction
        for correction in remaining_corrections:
            
            applied_corrections = best_combination + [correction]
     
            if task == "grammar":
                new_essay = apply_corrections_grammar(essay, applied_corrections)
            elif task == "longer":
                new_essay = apply_corrections_longer(essay, applied_corrections)
            
            new_score = get_essay_score(new_essay)

            print("New Score: ", new_score)
            
            score_improvement = new_score[0] - best_score[0]
            
            score_improvements.append((correction, score_improvement))
            
            # Print the output
            print("\nCorrection:")
           
            if task == "grammar":
       
                for i, item in enumerate(applied_corrections, start=1):
                    print(f"{i}. {item['type'].upper()} : ({item['original']} --> {item['correction']}) in sentence {item['sentence']}")
                    #yield f"data:{i}. {item['type'].upper()} : ({item['original']} --> {item['correction']}) in sentence {item['sentence']}\n\n"
         
            elif task == "longer":
       
                for i, item in enumerate(applied_corrections, start=1):
                    #yield f"data:{i}. {item['original']} --> {item['correction']}\n\n"
                    print(f"New Score: {new_score[1]} ({new_score[0]:.3f})")   
            
            if new_score[1] > original_score[1]:
                print("\nFinal score has exceeded the original score, exiting...")
                #yield f"data: Final score has exceeded the original score, exiting...\n\n"
          
                best_combination = best_combination + [correction]
                return best_combination, new_score, True  # Return immediately with the current correction
         
            elif new_score[0] > best_correction_score[0]:
                print("---Highest score improvement---")
                #yield f"data:---Highest score improvement---\n\n"
                # Update the best correction and best score for this iteration
                best_correction_score = new_score
                best_correction = correction
               
        # Sort the corrections based on score improvement
        score_improvements.sort(key=lambda x: x[1], reverse=True) 
    
        # If a new best correction was found, add it to the best combination
        if best_correction:
   
            best_combination.append(best_correction)
            best_score = best_correction_score
            # Remove the best correction from the remaining corrections
            remaining_corrections.remove(best_correction)
      
        else:
            print()
            print("=========== NO BETTER CORRECTION, ADDING CURRENT BEST ONE TO THE COMBINATION ==============")
            #yield f"data:=========== NO BETTER CORRECTION, ADDING CURRENT BEST ONE TO THE COMBINATION ==============\n\n\n"
            print()
            best_correction = score_improvements[0][0]
            best_combination.append(best_correction)
            remaining_corrections.remove(best_correction)
        
        print("Current best correction: ", best_correction)
        #yield f"data:Current best correction: {best_correction}\n\n"
        
    return best_combination, best_score, False


def get_counterfactuals(essay, corrections_grammar, corrections_longer):

    result = "Finding Counterfactuals . . . "
    #yield f"data: {result}\n\n"
    
    original_score = get_essay_score(essay)
    print(f"ORIGINAL SCORE : {original_score[1]} ({original_score[0]:.3f})")

    all_corrected_essay, all_correction_score = get_all_correction_score(essay, corrections_grammar)
    
    print(f"Score after applying all corrections: {all_correction_score[1]} ({all_correction_score[0]:.3f})")
    #yield f"data:Score after applying all corrections: {all_correction_score[1]} ({all_correction_score[0]:.3f})\n\n"
    
    if all_correction_score[1] > original_score[1]:
        print("Solution exist ... finding minimum change to improve score ...")
        #yield f"data:Solution exist ... finding minimum change to improve score ...\n\n"
    
        #minimum_change, best_score, status = yield from find_best_combination(essay, corrections_grammar, original_score)
        minimum_change, best_score, status = find_best_combination(essay, corrections_grammar, original_score)
        
        if status == True:
           
            print("\n=== Minimum Solution FOUND ===")
            #yield f"data: === Minimum Solution FOUND ===\n\n"
           
            print("\nMinimum changes to improve score:")
            #yield f"data: Minimum changes to improve score:\n\n"

            improvement = ""

            sentences = ""
                    
            for i, item in enumerate(minimum_change, start=1):
                print(f"{i}. {item['original']} --> {item['correction']}")

                improvement = improvement + item['original'] + ' --> ' + item['correction'] + ' [' + item['type'] + ']\n'
                sentence = sent_tokenize(essay)[item['sentence']-1]
                sentences = sentences + sentence + '\n'
                
                original = f"<span class='highlight-original'>{item['original']}</span>"
                corrected = f"<span class='highlight-correction'>{item['correction']}</span>"
                errortype = f"<span class='error-type'>[{item['type']}]</span>"
                arrow = f"<span class='arrow'> ➔ </span>"
                showing = errortype + original + arrow + corrected 
                #yield f"data: {i}. {showing} in sentence {item['sentence']}\n\n"
                print(f"{i}. {showing} in sentence {item['sentence']}\n")

            print("Improvement: ", improvement)
            print("Sentence: ", sentences)

            prompt = get_prompt_feedback(improvement, sentences)
            
            result = get_feedback_from_LLM(prompt)

            feedback = result[0]['html']

            feedback = f"<span class='highlight-original'>{feedback}</span>"
                        
            print(f"New Score: {best_score[1]} ({best_score[0]:.3f})")
            #yield f"data: New Score: {best_score[1]} ({best_score[0]:.3f})\n\n"
        
        else:
            print("\n=== Minimum Solution NOT FOUND, Must Apply All Corrections ===")
            print("\nChanges to improve score:")
            
            for i, item in enumerate(corrections_grammar, start=1):
                print(f"{i}. {item['type'].upper()} : ({item['original']} --> {item['correction']}) in sentence {item['sentence']}")
            print(f"\nNew Score: {all_correction_score[1]} ({all_correction_score[0]:.3f})")
            
    else:
        print("\n===All Corrections Applied, Solution NOT Found ===")
        print("Continue finding solution ..... ")
        #yield from finding_more_solution(all_corrected_essay, corrections_longer, original_score)
        finding_more_solution(all_corrected_essay, corrections_longer, original_score)

def finding_more_solution(essay, corrections, original_score):
    print("\nAll Corrected Essay:\n")
    #minimum_change, best_score, status = yield from find_best_combination(essay, corrections, original_score, "longer")
    minimum_change, best_score, status = find_best_combination(essay, corrections, original_score, "longer")
    
    if status == True:
       
        print("\n=== Minimum Solution FOUND (LONGER ESSAY) ===")
        #yield f"data: === Minimum Solution FOUND (LONGER ESSAY) ===\n\n"
       
        print("\nMinimum changes to improve score:")
        #yield f"data: Minimum changes to improve score\n\n"
        
        for i, item in enumerate(minimum_change, start=1):
            print(f"{i}. {item['original']} --> {item['correction']}")
            formatted_original = f"<div class='container'><span class='text-original'>{item['original']}</span><div class='arrow-down'></div></div>"    
            formatted_correction = f"<div class='container'><span class='text-correction'>{item['correction']}</span></div>"
            #yield f"data: {i}. Expand this sentence: {formatted_original}{formatted_correction}\n\n"
            print(f"{i}. Expand this sentence: {formatted_original}{formatted_correction}\n")
            #yield f"data: {i}. <span class='highlight-original'>{item['original']}</span> <div class='container'><div class='arrow-down'></div> <span class='highlight-correction'>{item['correction']}</span></div>\n\n"
            #yield f"data: {i}. <span class='highlight-original'>{item['original']}</span> <div class='container'><div class='arrow-down'></div> <span class='highlight-correction'>{item['correction']}</span></div>\n\n"
       
        print(f"New Score: {best_score[1]} ({best_score[0]:.3f})")
        #yield f"data: New Score: {best_score[1]} ({best_score[0]:.3f})"
    
    else:
        print("\n=== Minimum Solution NOT FOUND (LONGER ESSAY) ===")
        #yield f"data: === Minimum Solution NOT FOUND (LONGER ESSAY) ===\n\n"
        print("\nProgram terminated ... ")
    

def openai_request(openai_key, model_name, prompt):
    #one_prompt = instruction + essay
    
    client  = OpenAI(api_key=openai_key)
    
    messages = [
#         {
#             "role": "system",
#             "content": "You are ChatGPT, a large language model trained by OpenAI. You act as an english language expert that will provide corrections from any text. "
#         },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    #print("model: ", model_name)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        top_p=0.2,
        #n=1,
        max_tokens=2000,
        #response_format={ "type": "json_object" },
    )
    output = response.choices[0].message.content
    return output


def sanitize_string(data_str):
    # Escape all backslashes to ensure valid JSON parsing
    sanitized_str = data_str.encode('unicode_escape').decode('unicode_escape')
    
    # Remove or replace any problematic escape sequences
    sanitized_str = re.sub(r'\\x[0-9A-Fa-f]{2}', '', sanitized_str)  # Remove \x sequences
    sanitized_str = re.sub(r'\\u[0-9A-Fa-f]{4}', '', sanitized_str)  # Remove \u sequences (optional)

    # NEW: remove backslashes before invalid JSON escapes (e.g., \')
    # Keep only valid escapes: " \ / b f n r t u
    sanitized_str = re.sub(r'\\(?!["\\/bfnrtu])', r'', sanitized_str)

    return sanitized_str

    
def extract_json(data_str):

    # Sanitize the input string
    data_str = sanitize_string(data_str)
    
    # Try to find the JSON starting point by looking for the opening bracket
    json_start = data_str.find('[')
    json_end = data_str.rfind(']')  # Find the last closing bracket

    # If we can't find a JSON start or end, return an error or None
    if json_start == -1 or json_end == -1:
        raise ValueError("No JSON object could be decoded")

    # Extract the substring that is likely to be JSON
    potential_json_str = data_str[json_start:json_end + 1]

    try:
        # Try to parse the JSON
        json_data = json.loads(potential_json_str)
        return json_data
    except json.JSONDecodeError as e:
        # If JSON is not valid, raise an error
        raise ValueError("No JSON object could be decoded") from e

        
def get_essay_sentences(essay):
    # Tokenize the text into sentences
    sentences = sent_tokenize(essay)

    # Pair each sentence with its index
    indexed_sentences = [(index, sentence) for index, sentence in enumerate(sentences, start=1)]

    # Print the result
#     for index, sentence in indexed_sentences:
#         print(f"Sentence {index}: {sentence}")
        
    return indexed_sentences


def get_corrections_from_LLM_grammar(essay, model_name):
    print("Requesting grammar corrections from LLM ... ")

    sentences = get_essay_sentences(essay)

    prompt = get_prompt_grammar()

    prompt = prompt + str(sentences)
    result = openai_request(openai_key, model_name, prompt)
    print(result)
    result = extract_json(result)
    return result

def get_corrections_from_LLM_longer(essay, model_name, corrections_grammar):
    print("Requesting longer corrections from LLM ... ")

    corrected_essay = get_all_correction_score(essay, corrections_grammar)[0]
    
    sentences = get_essay_sentences(corrected_essay)

    prompt = get_prompt_longer()

    prompt = prompt + str(sentences)
    result = openai_request(openai_key, model_name, prompt)
    print(result)
    result = extract_json(result)
    return result



def get_prompt_grammar():
    prompt = """Identify the grammatical errors in this text. Please find at least five errors, but do not mention the entire phrase—focus only on the specific errors (maximum 3 consecutive words). Please give me the response in json format, which include: 
    1. "original": The original word or phrase.
    2. "correction": The corrected word or phrase.
    3. "type": The type of correction (e.g., "Spelling", "Grammatical", "Word Choice", "Other").
    4. "sentence": The index of the sentence where the correction should be applied.
 
    This is the only correct format of the response:
    [
      {
        "original": "their",
        "correction": "they're",
        "type": "Grammatical",
        "sentence": 4,
      }
    ]

    Text: \n"""
    return prompt

def get_prompt_longer():
    prompt = """From the provided text, please identify the three most important sentences. 
    Expand each of these sentences to double their original length.
    Write the expanded sentences using the style of an 8th grade student and easy-to-read language. 
    Please provide the response in json format, which include:

    1. "original": The original word or phrase.
    2. "correction": The corrected word or phrase.
    4. "sentence": The index of the sentence where the correction should be applied (starting from 1).

    This is the only correct format of the response:
    [
      {
        "original": "This is a book.",
        "correction": "Tt is crucial to emphasize that what we are currently examining is, without a doubt, a book—an item crafted from paper or other materials, often adorned with printed or handwritten content, serving the purpose of communicating ideas, stories, or information.",
        "sentence": [SENTENCE_NUMBER/INDEX STARTING FROM 1],
      }
    ]

    Text: \n"""
    return prompt

def get_prompt_feedback(improvement, sentence):
    prompt =  """\n
Can you give the explanation to the improvement as a feedback to students. Give the reason for every change. Please give your response using the following guideline:
1. Use html format, like bold or any style that will improve the presentation to the user 
2. Please use html list if there are multiple improved words or phrases
3. Enclosed the improved words or phrases in single quote
4. Do not mention the unchanged word and phrases
5. Please only answer with valid json format, without any additional text like this:
[
{"html": "[THE EXPLANATION HERE]"}
]
6. Make sure that the json only contain one "html" key.
"""
    return "Sentence: \n" + sentence + "\n\nImprovement: \n" + improvement + prompt


def get_feedback_from_LLM(prompt):
    print("\nRequesting feedback from LLM ... \n")
    model_name = 'gpt-4o-mini'
    result = openai_request(openai_key, model_name, prompt)
    #print(result)
    result = extract_json(result)
    return result