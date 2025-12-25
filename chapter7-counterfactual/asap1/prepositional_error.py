import pandas as pd
import numpy as np
import time
import json
import re
from openai import OpenAI
import joblib
from nltk.tokenize import sent_tokenize

def sanitize_string(data_str):
    # Escape all backslashes to ensure valid JSON parsing
    sanitized_str = data_str.encode('unicode_escape').decode('unicode_escape')
    
    # Remove or replace any problematic escape sequences
    sanitized_str = re.sub(r'\\x[0-9A-Fa-f]{2}', '', sanitized_str)  # Remove \x sequences
    sanitized_str = re.sub(r'\\u[0-9A-Fa-f]{4}', '', sanitized_str)  # Remove \u sequences (optional)

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


def get_prompt():
    prompt = """
You are an English grammar assistant. Your task is to detect **only prepositional errors** in the following essay. 
Ignore all other types of grammatical or spelling errors.

Please give the response in **JSON format**, where each item includes:
1. "original": The incorrect preposition.
2. "correction": The correct preposition.
3. "type": The type of correction (use "Preposition").
4. "sentence": The index (starting from 1) of the sentence where the correction should be applied.

This is the **only valid format** of the response:
[
  {{
    "original": "from",
    "correction": "of",
    "type": "Preposition",
    "sentence": 4
  }}
]

Now, analyze the following essay:
\n
"""
    return prompt


# Initialize OpenAI client
client = OpenAI(api_key="PUT_YOUR_OWN_KEY")  # Replace with your actual key

def openai_request(prompt, model="gpt-4o-mini"):
    """
    Sends an essay to ChatGPT and returns the list of preposition errors.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.2,
            #n=1,
            max_tokens=2000,
        )
        output = response.choices[0].message.content
        
        return output

    except Exception as e:
        print("Error:", str(e))
        return []

def get_prepositional_errors(df, essay_col="essay"):
    """
    Process all essays in a DataFrame and extract preposition errors.
    Adds two columns: 'prep_error_count' and 'prep_error_list'
    """
    prep_error_counts = []
    prep_error_lists = []

    start = time.time()

    for i, essay in enumerate(df[essay_col]):
        sentences = get_essay_sentences(essay)
        
        prompt = get_prompt() + str(sentences)
        
        corrections = openai_request(prompt)

        result = extract_json(corrections)
        
        prep_error_counts.append(len(result))
        prep_error_lists.append(result)

        joblib.dump(prep_error_lists, "preposition_error_lists")

        # Print progress every 10
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"Processed {i+1}/{len(df)} essays...")

    end = time.time()
    print(f"Done in {(end - start) / 60:.2f} minutes.")

    # Add to DataFrame
    df["prep_error_count"] = prep_error_counts
    df["prep_error_list"] = prep_error_lists

    return df