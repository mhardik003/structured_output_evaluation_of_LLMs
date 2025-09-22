import json
import re
from openai import OpenAI
import os
import time
from tqdm import tqdm

client = OpenAI(api_key="enter your api key here",
                base_url="https://llmproxy.stepsai.co/")


def extract_json_from_response(response_text):
    """
    Robust method to extract JSON from potentially messy LLM output.

    :param response_text: Full text response from the LLM
    :return: Parsed JSON or None if parsing fails
    """
    # Try to extract JSON from within ```json ``` code blocks
    json_code_block_match = re.search(
        r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
    if json_code_block_match:
        try:
            return json.loads(json_code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract JSON from within ``` ``` code blocks (no json specifier)
    code_block_match = re.search(
        r'```\s*({.*?})\s*```', response_text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try direct JSON parsing of the entire response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # If all parsing methods fail, return None
    return None


def evaluate_json_prediction(prediction, reference, rubric):
    """
    Evaluate a JSON prediction against a reference using a detailed rubric.

    :param prediction: The JSON prediction to evaluate
    :param reference: The reference JSON to compare against
    :param rubric: The evaluation criteria
    :return: Dictionary containing evaluation scores and rationales
    """

    # Define the expected output structure
    output_structure = {
        "criteria_scores": {
            "semantic_equivalence": {
                "score": "int between 1-5",
                "rationale": "string explaining the scoring"
            },
            "structural_hierarchy": {
                "score": "int between 1-5",
                "rationale": "string explaining the scoring"
            },
            "completeness": {
                "score": "int between 1-5",
                "rationale": "string explaining the scoring"
            },
            "data_consistency": {
                "score": "int between 1-5",
                "rationale": "string explaining the scoring"
            }
        }
    }

    # Construct a detailed prompt for the AI evaluator
    prompt = f"""
    You are an expert JSON evaluator. Your task is to systematically evaluate a JSON prediction against a reference JSON 
    using the following detailed rubric:

    {rubric}

    IMPORTANT: 
    1. Return the evaluation as a JSON wrapped in ```json ``` code blocks
    2. Follow this EXACT JSON structure:

    {json.dumps(output_structure, indent=2)}

    REFERENCE JSON:
    {json.dumps(reference, indent=2)}

    PREDICTION JSON:
    {json.dumps(prediction, indent=2)}

    Evaluation Instructions:
    1. Score each criterion on a scale of 1-5
    2. Provide a clear rationale for each score
    3. Ensure the response is a valid JSON object that exactly matches the structure above
    """

    # Make API call to get the evaluation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert JSON evaluation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )

    # Get the response text
    response_text = response.choices[0].message.content

    # Attempt to parse the JSON
    evaluation = extract_json_from_response(response_text)

    # Return the evaluation or None if parsing fails
    return evaluation


def main():
    # Example rubric (the one you provided)
    rubric = """
# Evaluation Rubric

# Criterion 1: Semantic Equivalence to the Reference

## Score 5

The prediction is fully semantically equivalent to the reference. All keys, values, and relationships convey the exact meaning as defined in the reference without any deviation.

## Score 4

The prediction is largely semantically equivalent to the reference. Minor differences (e.g., alternative but equivalent key names) exist, but they do not alter the intended meaning.

## Score 3

The core semantic content is generally equivalent, yet there are noticeable differences that may require interpretation. Some elements are only partially equivalent to the reference.

## Score 2

Several semantic elements are not equivalent to the reference. Key concepts are misrepresented or omitted, leading to a significant deviation in intended meaning.

## Score 1

The prediction fails to capture the intended meaning of the reference. Critical semantic elements are missing or grossly misinterpreted, resulting in a lack of equivalence.

# Criterion 2: Structural Hierarchy and Organization

## Score 5

The JSON is organized into a clear, logical hierarchy that mirrors the conceptual model of the reference. Nested elements are intuitively grouped to reflect relationships.

## Score 4

The hierarchy is mostly well-defined with only minor areas where grouping could be refined. The overall organization closely reflects the reference.

## Score 3

A hierarchical structure is present but is uneven or partially flat. Some logical groupings exist but could be significantly improved to align with the reference.

## Score 2

The structure is weakly hierarchical or overly flat, with minimal logical grouping. It does not clearly represent the reference’s intended structure.

## Score 1

There is no discernible hierarchy. The JSON is disorganized, and elements are arbitrarily placed, failing to reflect the reference’s structure.

# Criterion 3: Completeness

## Score 5

Every key, nested element, and data point present in the reference is included in the prediction, with no omissions or extraneous additions.

## Score 4

The prediction is nearly complete. There might be one or two trivial omissions or minor redundancies that do not impact the overall representation.

## Score 3

Some elements are missing or misplaced relative to the reference. While the core information is present, gaps exist that may affect full comprehension.

## Score 2

A significant portion of keys or nested elements are missing or incorrectly placed, which could lead to misinterpretation of the data.

## Score 1

The prediction is largely incomplete, with most of the reference’s elements missing or incorrectly represented.

# Criterion 4: Data Consistency and Correctness

## Score 5
All data types, key naming conventions, and value formats are consistent and match the reference exactly, ensuring accurate interpretation.

## Score 4

Minor inconsistencies in data types or naming exist, but they do not materially affect the data’s overall correctness or interpretation.

## Score 3

Some inconsistencies in data types or formats are present, which may cause occasional confusion when compared with the reference.

## Score 2

Frequent inconsistencies are evident, with notable mismatches in data types or formats that detract from the overall correctness.

## Score 1

The prediction exhibits significant errors and inconsistencies in data representation, severely impairing the accurate interpretation of the information.
    """
    with open("cleaned.jsonl") as f:
        total = sum(1 for _ in f)

    start_from = 0
    with open("ai_evaluation.jsonl") as f:
        for line in f:
            start_from += 1

    with open("cleaned.jsonl") as f, open("ai_evaluation.jsonl", "a") as fout:
        for i, line in tqdm(enumerate(f), total=total):
            if i < start_from:
                continue
            data = json.loads(line)
            phi_prediction = data["phi"]
            qwen_prediction = data["qwen"]
            reference = data["reference"]
            phi_evaluation = evaluate_json_prediction(
                phi_prediction, reference, rubric)
            qwen_evaluation = evaluate_json_prediction(
                qwen_prediction, reference, rubric
            )
            out = {**data, "phi_evaluation": phi_evaluation,
                   "qwen_evaluation": qwen_evaluation}
            print(json.dumps(out), file=fout)
            fout.flush()
            time.sleep(0.2)


if __name__ == "__main__":
    main()
