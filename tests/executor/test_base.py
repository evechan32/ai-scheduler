import pytest
from moesim.executor.base import ExpertExecutor


def test_executor_abstract():
    ex = ExpertExecutor()
    with pytest.raises(NotImplementedError):
        ex.load(["e0"])
    with pytest.raises(NotImplementedError):
        ex.execute_gpu("e0", object())
