import pytest

from moesim.executor.backends.vllm import VLLMExecutor  # noqa: E402
from moesim.executor.backends.llama_cpp import LlamaCppExecutor  # noqa: E402


class FakeVLLM:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(("generate", args, kwargs))
        return "output"


class FakeLlama:
    def __init__(self):
        self.calls = []

    def eval(self, *args, **kwargs):
        self.calls.append(("eval", args, kwargs))
        return [0.1, 0.2]


def test_vllm_executor_records_residency_and_forwards():
    engine = FakeVLLM()
    ex = VLLMExecutor(engine)
    ex.load(["e0"])
    assert ex.residency["e0"] == "loaded"
    out = ex.execute_gpu("e0", object())
    assert out == "output"
    assert engine.calls[-1][0] == "generate"
    ex.unload(["e0"])
    assert ex.residency["e0"] == "unloaded"


def test_llamacpp_executor_forwards_to_eval():
    engine = FakeLlama()
    ex = LlamaCppExecutor(engine)
    ex.load(["e1"])
    assert ex.residency["e1"] == "loaded"
    out = ex.execute_gpu("e1", [1, 2, 3])
    assert out == [0.1, 0.2]
    assert engine.calls[-1][0] == "eval"
