"""
llm_handler.py
--------------
Abstracts "which local LLM backend are we using" behind a single
get_llm() factory. This is what lets the rest of the pipeline stay
backend-agnostic: swap HF <-> Ollama by changing one config value,
no changes needed in rag_pipeline.py.
"""

from langchain_core.language_models import BaseChatModel, LLM

from src import config


def _load_huggingface_llm() -> LLM:
    """
    Load a small instruction-tuned model locally via transformers and
    wrap it as a LangChain LLM through HuggingFacePipeline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.HF_MODEL_NAME,
        device_map="auto",       # uses GPU if available, else CPU
        trust_remote_code=True,
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.HF_MAX_NEW_TOKENS,
        temperature=config.HF_TEMPERATURE,
        do_sample=config.HF_TEMPERATURE > 0,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def _load_ollama_llm() -> BaseChatModel:
    """
    Load a locally-running Ollama model (requires `ollama serve` running
    and the model already pulled, e.g. `ollama pull llama3`).
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.OLLAMA_MODEL_NAME, temperature=config.OLLAMA_TEMPERATURE)


def get_llm():
    """
    Factory returning a ready-to-use local LLM object based on
    config.LLM_BACKEND ("huggingface" or "ollama").
    """
    if config.LLM_BACKEND == "huggingface":
        print(f"Loading local Hugging Face model: {config.HF_MODEL_NAME} ...")
        return _load_huggingface_llm()
    elif config.LLM_BACKEND == "ollama":
        print(f"Connecting to local Ollama model: {config.OLLAMA_MODEL_NAME} ...")
        return _load_ollama_llm()
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'. "
            "Use 'huggingface' or 'ollama'."
        )
