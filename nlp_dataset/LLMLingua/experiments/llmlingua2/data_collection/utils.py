from time import sleep

from transformers import AutoTokenizer
from openai import OpenAI
import tiktoken


def query_llm(
    prompt,
    model,
    model_name,
    max_tokens,
    tokenizer=None,
    chat_completion=False,
    **kwargs,
):
    SLEEP_TIME_FAILED = 62

    request = {
        "temperature": kwargs["temperature"] if "temperature" in kwargs else 0.0,
        "top_p": kwargs["top_p"] if "top_p" in kwargs else 1.0,
        "seed": kwargs["seed"] if "seed" in kwargs else 42,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    if chat_completion:
        request["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        request["prompt"] = prompt

    answer = None
    response = None
    while answer is None:
        try:
            response = model.create(engine=model_name, **request)
            answer = (
                response["choices"][0]["message"]["content"]
                if chat_completion
                else response["choices"][0]["text"]
            )
        except Exception as e:
            answer = None
            print(f"error: {e}, response: {response}")
            sleep(SLEEP_TIME_FAILED)
    # sleep(SLEEP_TIME_SUCCESS)
    return answer


def load_model_and_tokenizer(model_name_or_path, chat_completion=False):
    if model_name_or_path == "gpt-4o":
        client = OpenAI(organization="XXXXXXX")
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    else:   # NOTE: using a hugging face model requires that the model is hosted and compatible with the OpenAI API (e.g., vLLM)
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    model = client.chat.completions
    return model, tokenizer
