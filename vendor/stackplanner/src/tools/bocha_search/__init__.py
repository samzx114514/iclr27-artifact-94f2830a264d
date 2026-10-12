from .web_searcher import WebSearcher
from .web_search_utils import create_web_searcher
try:  # optional: offline retrieval-quality helper that needs ragas
    from .retrieval_quality_evaluator import RetrievalQualityEvaluator
except ModuleNotFoundError:  # pragma: no cover - deployment without ragas
    RetrievalQualityEvaluator = None
