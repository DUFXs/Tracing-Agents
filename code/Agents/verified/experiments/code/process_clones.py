import os
import json
import pandas as pd
import re
from collections import defaultdict
import black
import litellm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
import time
# Global variable for Vertex AI credential
VERTEX_AI_CREDENTIAL = ""

# Load evaluation results and filter resolved instances
def load_and_filter_results(json_file):
    with open(json_file, "r") as f:
        results = json.load(f)
    
    filtered_results = []
    
    for difficulty, instances in results.items():
        for instance_id, agents in instances.items():
            for agent, data in agents.items():
                if data.get("resolved") is True:  # Filter where resolved is True
                    ground_truth_patches = data.get("ground_truth_patch", {})
                    agent_patches = data.get("agent_patch", {})

                    # for agent_file, agent_patch in agent_patches.items():
                    #     for gt_file, gt_patch in ground_truth_patches.items():
                    # filtered_results.append({
                    #     "instance_id": instance_id,
                    #     "agent": agent,
                    #     "agent_patch_file": agent_file,
                    #     "agent_patch": agent_patches,
                    #     "ground_truth_patch_file": gt_file,
                    #     "ground_truth_patch": gt_patch
                    # })

                    filtered_results.append({
                    "instance_id": instance_id,
                    "agent": agent,
                    "agent_patch": agent_patches,
                    "ground_truth_patch": ground_truth_patches
                    })
    
    return filtered_results


# Query Litellm Bedrock API
def query_litellm(prompt, model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.7):
    """
    Queries the Litellm Bedrock API and returns the response.
    """
    try:
        response = litellm.completion(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. LIMIT RESPONSE TO 130 TOKENS."},
                {"role": "user", "content": prompt},
            ],
            vertex_credentials=VERTEX_AI_CREDENTIAL,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API query error: {e}")
        return None
    
def query_openai_o1(prompt, model="o1", temperature=0.7):
    client = openai
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Limit your response to 500 tokens."},
            {"role": "user", "content": prompt},
        ])
   
    return response.choices[0].message.content

def query_openai_4omini(prompt, model="gpt-4o-mini", temperature=0.7):
    print("Querying...")
    client = openai
    print("Here...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Limit your response to 500 tokens."},
            {"role": "user", "content": prompt},
        ])
    print("but...?")
    print("Response...., ",response.choices[0].message.content)
    return response.choices[0].message.content
# Generate prompt for clone detection
def generate_prompt(data):
    return f"""###Instruction
You are a code-clone detection assistant. You will be given two examples from a GitHub patch, both examples
in the same language. You have to determine whether they are clones of each other.

Determine whether they are clones of each other based on the following definitions:
- Type-1 Clone: Identical except for differences in whitespace, layout, and comments.
- Type-2 Clone: Identical except for differences in identifier names, literal values, whitespace, layout, and comments.
- Type-3 Clone: Syntactically similar with added, modified, or removed statements.
- Type-4 Clone: Syntactically dissimilar but functionally equivalent.

###Input
Are these code samples clones?

<Code sample 1 start>
Patch: {data["agent_patch"]}
<Code sample 1 end>

<Code sample 2 start>
Patch: {data["ground_truth_patch"]}
<Code sample 2 end>

###Response 
Return your response and reasoning in one of these formats: 

{{"is_clone": true, "classification": "type-1", "reasoning": "These code samples are type-1 clones because.."}} 
{{"is_clone": true, "classification": "type-2", "reasoning": "These code samples are type-2 clones because.."}} 
{{"is_clone": true, "classification": "type-3", "reasoning": "These code samples are type-3 clones because.."}} 
{{"is_clone": true, "classification": "type-4", "reasoning": "These code samples are type-4 clones because.."}} 

or

{{"is_clone": false, "classification": "N/A", "reasoning": "These code samples are not lones because.."}}  

"""

def process_entry(entry, output_dir, model):
        print("Generating prompt...")
        prompt = generate_prompt(entry)

        output_file = os.path.join(output_dir, f"clone_detection_result_{entry['instance_id']}_{entry['agent']}_{model}.json")
        print("Checked before querying!")
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping query.")
            return
        
        response =  query_litellm(prompt) # query_openai_4omini(prompt, model, 0.7)
        if response:
            result = {
                "instance_id": entry["instance_id"],
                "agent": entry["agent"],
                "agent_patch": entry["agent_patch"],
                "ground_truth_patch": entry["ground_truth_patch"],
                "response": response
            }
            print("Trying to save....")
            save_response(result, output_dir, "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

# def process_clones(filtered_results, model="gpt-4o-mini", temperature=0.7): 
    
#     responses = []

#     def process_entry(entry, output_dir):
#         prompt = generate_prompt(entry)
#         response = query_openai_o1(prompt, model, temperature)
#         if response:
#             result = {
#                 "instance_id": entry["instance_id"],
#                 "agent": entry["agent"],
#                 "agent_patch_file": entry["agent_patch_file"],
#                 "ground_truth_patch_file": entry["ground_truth_patch_file"],
#                 "response": response
#             }
#             save_response(result, output_dir, model)

    # with ThreadPoolExecutor(max_workers=50) as executor:
    #     futures = {executor.submit(process_entry, entry): entry for entry in filtered_results}

    #     for future in as_completed(futures):
    #         result = future.result()
    #         if result:
    #             responses.append(result)
    # return responses  # Ensure this is outside the loop
    
def process_clones(filtered_results, model="gpt-4o-mini", temperature=0.7, output_dir="results"): 
    tasks = []
    num = 0
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for entry in filtered_results:
            print("Processing entry: ", num)
            num += 1
            future = executor.submit(process_entry, entry, output_dir, model)
            tasks.append(future)
            time.sleep(0.55)

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred while processing an entry: {e}")


# def save_response(result, output_dir="individual_results", model):
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     output_file = os.path.join(output_dir, f"clone_detection_result_{result['instance_id']}_{result['agent']}_{model}.json")
#     with open(output_file, "w") as f:
#         json.dump(result, f, indent=4)
#     print(f"Response for {result['instance_id']} saved to {output_file}")

def save_response(result, output_dir, model):
    print("Got to saving response...")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"clone_detection_result_{result['instance_id']}_{result['agent']}_{model}.json")
    
    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping write.")
        return
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Response for {result['instance_id']} saved to {output_file}")


def save_responses(responses, output_dir="results"): 
    os.makedirs(output_dir, exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") 
    output_file = os.path.join(output_dir, f"clone_detection_results_50_mw_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(responses, f, indent=4)

    print(f"Responses saved to {output_file}")


filtered_data = load_and_filter_results("evaluation_results.json")

if not filtered_data:
    print("No resolved instances found.")
else:
    print(f"Processing {len(filtered_data)} resolved instances...")
    #clone_detection_results = process_clones(filtered_data)
    #save_responses(clone_detection_results)
    process_clones(filtered_data, output_dir="individual_results")
