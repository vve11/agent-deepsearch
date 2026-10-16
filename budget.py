"""异步任务独立的预算上下文；计数由 SQLite 持久化。"""
from contextvars import ContextVar
from contextlib import contextmanager


class BudgetExceeded(ValueError):
    pass


_current = ContextVar("research_budget", default=None)


@contextmanager
def use_budget(run_id, model_name):
    token = _current.set((run_id, model_name))
    try:
        yield
    finally:
        _current.reset(token)


def reserve(kind):
    state = _current.get()
    if state is not None:
        from pipeline_storage import reserve_budget
        reserve_budget(state[0], kind)


def current_model(default):
    state = _current.get()
    return state[1] if state is not None else default


def tools_available():
    state = _current.get()
    if state is None:
        return True
    from pipeline_storage import load_pipeline
    budget = load_pipeline(state[0])
    return budget['tool_used'] < budget['tool_limit']
