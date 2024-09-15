import openai
import json
import os
import time

def load_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return api_key

# Initialize the OpenAI client
client = openai.OpenAI(api_key=load_api_key())



def evaluate_response(prompt_data, model="gpt-3.5-turbo", max_retries=5):
    prompt = prompt_data.get("prompt", "").strip()
    response = prompt_data.get("response", "").strip()

    if not prompt or not response:
        return {"error": "Prompt or response missing from input data"}

    evaluation_prompt = (
        "Evaluate the following response to the given prompt.\n\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n\n"
        "Provide three integer scores between 1 and 100 separated by spaces for the following criteria:\n"
        "1. Quality: How well the response answers the prompt.\n"
        "2. Complexity: How detailed and sophisticated the response is.\n"
        "3. Accuracy: How factually correct the response is.\n\n"
        "Only reply with three integer numbers separated by spaces. Example: '85 90 80'"
    )

    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that evaluates responses based on given criteria."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=10,
                temperature=0
            )

            evaluation_text = completion.choices[0].message.content.strip()
            
            # Validate the response format
            scores = evaluation_text.split()
            if len(scores) != 3 or not all(score.isdigit() and 1 <= int(score) <= 100 for score in scores):
                return {"error": f"Invalid evaluation format: '{evaluation_text}'"}

            # Convert scores to integers
            scores = list(map(int, scores))
            return {
                "Quality": scores[0],
                "Complexity": scores[1],
                "Accuracy": scores[2]
            }

        except openai.RateLimitError:
            wait_time = 2 ** retries
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
        except openai.OpenAIError as e:
            return {"error": str(e)}

    return {"error": "Max retries exceeded."}


def record_evaluation(input_file, output_file, model="gpt-3.5-turbo"):
    """
    Reads prompt-response pairs from an input JSON file, evaluates them, and writes the results to an output JSON file.

    Parameters:
        input_file (str): Path to the input JSON file containing a list of prompt-response pairs.
        output_file (str): Path to the output JSON file to store evaluations.
        model (str): The OpenAI model to use for evaluation.
    """
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    with open(input_file, "r") as infile:
        try:
            data = json.load(infile)
            if not isinstance(data, list):
                raise ValueError("Input JSON should be a list of prompt-response pairs.")
        except json.JSONDecodeError as jde:
            print(f"Error decoding JSON: {jde}")
            return
        except ValueError as ve:
            print(f"Invalid input format: {ve}")
            return

    evaluations = []
    for idx, item in enumerate(data, start=1):
        print(f"Evaluating item {idx}/{len(data)}...")
        evaluation = evaluate_response(item, model=model)
        evaluations.append({
            "prompt": item.get("prompt", ""),
            "response": item.get("response", ""),
            "evaluation": evaluation
        })

    with open(output_file, "w") as outfile:
        json.dump(evaluations, outfile, indent=4)
    print(f"Evaluations recorded in '{output_file}'.")


if __name__ == "__main__":
    input_json_file = "input_prompt.json"    # Ensure this file exists and contains a list of prompt-response pairs
    output_json_file = "evaluation_output.json"
    record_evaluation(input_json_file, output_json_file)
