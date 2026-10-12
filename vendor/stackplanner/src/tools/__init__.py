"""Tool exports loaded on demand so optional search dependencies stay optional."""

from importlib import import_module
from typing import Any

__all__ = [
    "crawl_tool",
    "python_repl_tool",
    "get_web_search_tool",
    "search_docs_tool",
    "get_retriever_tool",
    "VolcengineTTS",
]


_EXPORTS = {
    "crawl_tool": (".crawl", "crawl_tool"),
    "python_repl_tool": (".python_repl", "python_repl_tool"),
    "get_web_search_tool": (".search", "get_web_search_tool"),
    "search_docs_tool": (".get_docs_info", "search_docs_tool"),
    "get_retriever_tool": (".retriever", "get_retriever_tool"),
    "VolcengineTTS": (".tts", "VolcengineTTS"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
