# This script provides the reusable inference helpers. run_batch_inference_local loads a chat model from a
# directory, sends it the same system prompt with each user message, and returns every question paired with
# the model's completion; answers are decoded greedily, so results are reproducible, and generated with continuous
# batching, so a finished answer frees its slot for the next question instead of waiting on the longest one.
# run_inference_from_openrouter_api does the same through the OpenRouter API, sending many requests concurrently
# with the model's native reasoning enabled, optionally routed to preferred providers first, and keeps only the
# visible completion. Each request gets a hard 120 second deadline and is retried on timeouts, connection errors,
# rate limits and server errors.

import asyncio
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.configuration_utils import ContinuousBatchingConfig

from scripts.inference.models import Inference


def run_batch_inference_local(
    model_directory: Path,
    system_prompt: str,
    user_messages: list[str],
    max_new_tokens: int = 4096,
    max_kv_cache_memory_fraction: float = 0.9,
) -> list[Inference]:
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype=torch.bfloat16, device_map="cuda").eval()

    completions = generate_greedy_completions_with_continuous_batching(
        model,
        tokenizer,
        system_prompt,
        user_messages,
        max_new_tokens,
        max_kv_cache_memory_fraction,
    )
    return [
        Inference(question=user_message, answer=completion)
        for user_message, completion in zip(user_messages, completions, strict=True)
    ]


def generate_greedy_completions_with_continuous_batching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    user_messages: list[str],
    max_new_tokens: int,
    max_kv_cache_memory_fraction: float,
) -> list[str]:
    prompt_token_ids = [
        tokenizer(build_chat_prompt(tokenizer, system_prompt, user_message))["input_ids"] for user_message in user_messages
    ]
    greedy_generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    )
    continuous_batching_config = ContinuousBatchingConfig(max_memory_percent=max_kv_cache_memory_fraction)

    generation_outputs = model.generate_batch(
        prompt_token_ids,
        generation_config=greedy_generation_config,
        continuous_batching_config=continuous_batching_config,
        progress_bar=True,
    )

    failed_generation_errors = [
        generation_output.error for generation_output in generation_outputs.values() if generation_output.error is not None
    ]
    if failed_generation_errors:
        raise RuntimeError(
            f"Continuous batching failed for {len(failed_generation_errors)} requests: {failed_generation_errors[0]}"
        )
    return [
        tokenizer.decode(generation_outputs[f"req_{request_index}"].generated_tokens, skip_special_tokens=True)
        for request_index in range(len(user_messages))
    ]


def build_chat_prompt(tokenizer: PreTrainedTokenizerBase, system_prompt: str, user_message: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_inference_from_openrouter_api(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_messages: list[str],
    preferred_providers: tuple[str, ...] = (),
    reasoning_effort: str = "high",
    max_concurrent_requests: int = 64,
) -> list[Inference]:
    openrouter_options = {
        "reasoning": {"effort": reasoning_effort},
        "provider": {"order": list(preferred_providers), "allow_fallbacks": True},
    }
    completions = asyncio.run(
        request_openrouter_completions(
            api_key,
            model_name,
            system_prompt,
            user_messages,
            openrouter_options,
            max_concurrent_requests,
        )
    )
    return [
        Inference(question=user_message, answer=completion)
        for user_message, completion in zip(user_messages, completions, strict=True)
    ]


async def request_openrouter_completions(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_messages: list[str],
    openrouter_options: dict,
    max_concurrent_requests: int,
) -> list[str]:
    concurrency_limit = asyncio.Semaphore(max_concurrent_requests)

    async with AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, max_retries=0) as openrouter_client:
        completion_requests = [
            request_openrouter_completion(
                openrouter_client,
                concurrency_limit,
                model_name,
                system_prompt,
                openrouter_options,
                user_message,
            )
            for user_message in user_messages
        ]
        return await tqdm_asyncio.gather(*completion_requests)


@retry(
    retry=retry_if_exception_type(
        (TimeoutError, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
    ),
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(max=30),
    reraise=True,
)
async def request_openrouter_completion(
    openrouter_client: AsyncOpenAI,
    concurrency_limit: asyncio.Semaphore,
    model_name: str,
    system_prompt: str,
    openrouter_options: dict,
    user_message: str,
) -> str:
    async with concurrency_limit:
        chat_completion_request = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=8192,
            extra_body=openrouter_options,
        )
        chat_completion = await asyncio.wait_for(chat_completion_request, timeout=120)
    return chat_completion.choices[0].message.content or ""
