import openai
import re
import json
import asyncio
import time
from datasets import load_dataset
from tqdm import tqdm
from chonkie import SentenceChunker


dataset = load_dataset("wikipedia", "20220301.en",
                       trust_remote_code=True, num_proc=8)

client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="abcd",
    max_retries=1,
    timeout=1200
)

chunker = SentenceChunker(
    # Supports string identifiers
    tokenizer_or_token_counter="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
    chunk_size=5000,                  # Maximum tokens per chunk
    chunk_overlap=0,               # Overlap between chunks
    min_sentences_per_chunk=1        # Minimum sentences in each chunk
)


BATCH_SIZE = 1  # Adjust the batch size as needed
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"


def create_prompt(text: str) -> str:
    """
    Creates a prompt instructing the model to convert the unstructured text into
    a deeply nested, semantically rich JSON object. The output must be enclosed within
    triple backticks. It may or may not include the 'json' language tag.
    """
    prompt = f"""You are an expert in semantic analysis and structured data conversion. Your task is to transform the following unstructured text into a deeply nested, semantically rich JSON object that captures every detail, nuance, and implicit context from the original input.

Always Follow:
- **Complete Detail Capture:** Ensure no piece of information—explicit or implicit—is omitted.
- **Deep Hierarchical Structuring:** Organize the output into nested objects and arrays that accurately reflect the relationships and hierarchies inherent in the text.
- **Semantic Clarity:** Always use descriptive and meaningful key names that are present in the text and clearly represent the nature of the data (e.g., "personal_information", "achievements"). For any ambiguous or context-dependent details, include clarifications or notes within the structure.
- **Preciseness** -  Do not use "type" or "description" as keys, prefer more in-context keys like related to the broader category. Make sure the keys are as specific as they can get.
- **Valid JSON Output:** Your final output must be syntactically correct JSON.
- **Standardized Formatting:** Enclose the JSON output within triple backticks. 

Unstructured Text:
{text}

Please output only the JSON object, enclosed in triple backticks and try to minimise the use of type, description, category and other vague names, use specific keys."""
    return prompt


def extract_json_from_response(response_text: str) -> dict:
    """
    Extracts the JSON block from the model's response.
    The JSON must be enclosed in triple backticks. This regex handles cases where the language tag
    (e.g., 'json') may or may not be present after the opening backticks.
    """
    # Regex pattern: match triple backticks with an optional language specifier, then capture content until closing backticks.
    pattern = r"```(?:\w+)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return None
    else:
        print("No JSON block found in the response.")
        return None


async def call_model(prompt: str) -> str:
    """
    Calls the OpenAI ChatCompletion API with the provided prompt.
    """
    response = await client.chat.completions.create(
        model=MODEL_NAME,  # or any other model of your choice
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


async def process_text(text: str) -> dict:
    """
    Processes a single text entry: creates a prompt, calls the model,
    and extracts the JSON output.
    """
    prompt = create_prompt(text)
    response_text = await call_model(prompt)
    parsed_json = extract_json_from_response(response_text)
    return parsed_json


async def main():
    start_from = 0
    try:
        with open("structured.jsonl") as f:
            start_from = len(f.readlines())
    except:
        pass
    print("starting from", start_from)

    with open("structured.jsonl", "a") as f:
        # Process texts in batches of BATCH_SIZE
        bar = tqdm(total=len(dataset["train"]) - start_from)
        for i in range(start_from, len(dataset["train"]), BATCH_SIZE):
            batch = [chunker.chunk(dataset["train"][j]['text'])[0].text
                     for j in range(i, i+BATCH_SIZE)]
            tasks = [process_text(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)
            for j, (r, text) in enumerate(zip(batch_results, batch)):
                print(json.dumps({"input": text, "output": r}), file=f)
                f.flush()
            bar.update(n=BATCH_SIZE)


if __name__ == "__main__":
    asyncio.run(main())
