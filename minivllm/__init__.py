"""Educational reimplementation of vLLM-style inference for text-only Gemma 3."""
from minivllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from minivllm.llm import LLM
        return LLM
    raise AttributeError(name)
