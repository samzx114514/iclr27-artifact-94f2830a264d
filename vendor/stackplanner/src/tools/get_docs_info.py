import os

import requests
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from .decorators import log_io
from src.utils.logger import logger
from src.utils.reference_utils import global_reference_map

# RAGFlow knowledge-base service backing the research document-retrieval tool.
# Point these at your own deployment; no endpoint is bundled with the release.
RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:8509")
RAGFLOW_TENANT_ID = os.environ.get("RAGFLOW_TENANT_ID", "")
RAGFLOW_OWNER_IDS = [
    item.strip()
    for item in os.environ.get("RAGFLOW_OWNER_IDS", "").split(",")
    if item.strip()
]
RAGFLOW_KB_NAME = os.environ.get("RAGFLOW_KB_NAME", "default")


def get_kb_id_by_name(kb_name):
    url = f"{RAGFLOW_BASE_URL}/v1/kb_api/list"
    params = {
        "page": 1,
        "page_size": 100,
        "keywords": kb_name,
        "orderby": "create_time",
        "desc": "true",
    }
    data = {
        "tenant_id": RAGFLOW_TENANT_ID,
        "owner_ids": RAGFLOW_OWNER_IDS,
    }

    try:
        resp = requests.post(url, params=params, json=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            logger.error(f"获取知识库列表失败: {result}")
            return None

        for kb in result.get("data", {}).get("kbs", []):
            if kb.get("name") == kb_name:
                return kb.get("id")

        logger.warning(f"未找到知识库: {kb_name}")
        return None

    except requests.RequestException as e:
        logger.error(f"请求知识库列表接口异常: {e}")
        return None
    except Exception as e:
        logger.error(f"解析知识库列表异常: {e}")
        return None


# 新接口
def search_docs(question, top_k=5, config: RunnableConfig = None):
    docs = []
    if config == None:
        knowledge_base_name = RAGFLOW_KB_NAME
        logger.info("knowledge_base_name使用的默认参数")
    else:
        knowledge_base_name = config["configurable"]["knowledge_base_name"]
        logger.info("knowledge_base_name使用的自定义参数")

    kb_id = get_kb_id_by_name(knowledge_base_name)
    if kb_id == None:
        logger.error(f"知识库 {knowledge_base_name}不存在")
        return docs
    logger.info(f"knowledge_base_name: {knowledge_base_name}")

    api_url = f"{RAGFLOW_BASE_URL}/v1/chunk_api/retrieval_test"

    query = {
        "tenant_id": RAGFLOW_TENANT_ID,
        "owner_ids": RAGFLOW_OWNER_IDS,
        "kb_id": [kb_id],
        "similarity_threshold": 0.3,  # 相似度阈值
        "question": question,
        "page": 1,
        "size": top_k,
    }
    try:
        response = requests.post(api_url, json=query)
        # logger.info(f"response: {response}")
        if response.status_code == 200:
            results = response.json()
            # logger.info(f"results: {results}")

            # results 去重
            chunks = results.get("data", {}).get("chunks", [])
            seen = set()
            for chunk in chunks:
                source = chunk.get("docnm_kwd", "")
                content = chunk.get("content_with_weight", "")

                if not source or not content:
                    continue

                # 去重依据：文档名 + 内容
                identifier = (source, content)
                if identifier in seen:
                    continue
                seen.add(identifier)

                docs.append({"source": source, "content": content})
        else:
            logger.error(
                f"请求失败，状态码: {response.status_code}，错误信息: {response.text}"
            )
        return docs
    except requests.RequestException as e:
        logger.error(f"请求过程中出现异常: {e}")
        return docs


@tool
@log_io
def search_docs_tool(
    question: Annotated[str, "检索的问题，使用语义相似度匹配"], config: RunnableConfig
) -> dict:
    """
    使用这个工具查询本地存储的领域知识库，检索方式为语义相似度匹配，返回与question相关的文档内容。
    """
    logger.debug(f"config:{config}")
    if config != None:
        session_id = config["configurable"]["thread_id"]
    else:
        session_id = None

    docs = search_docs(question, 15, config)
    # return {"query": question, "docs": docs}

    if session_id is None:
        logger.error("session_id is None in config")
        return {"query": question, "docs": docs}
    # logger.debug(f"检索到的文档{docs}")
    ids = global_reference_map.add_references(session_id, docs)
    if not ids or not docs:
        logger.warning("ids or docs is empty, returning empty result")
        return {"query": question, "docs": []}
    # 先把docs按ids升序排序
    # ["【文档x】name\ncontent\n",...]
    ids, docs = zip(*sorted(zip(ids, docs)))
    # rename_docs = ["【文档" + str(doc_id) + "】" + doc.get("source", "") + "\n" + doc.get("content", "") for doc_id, doc in zip(ids, docs)]
    rename_docs = [
        {
            "source": "【文档" + str(doc_id) + "】" + doc.get("source", ""),
            "content": doc.get("content", ""),
        }
        for doc_id, doc in zip(ids, docs)
    ]
    return {"query": question, "docs": rename_docs}


def search_docs_with_ref(
    question: Annotated[str, "检索的问题，使用语义相似度匹配"],
    top_k,
    config: RunnableConfig,
) -> dict:
    """
    使用这个工具查询本地存储的领域知识库，检索方式为语义相似度匹配，返回与question相关的文档内容。
    """
    logger.debug(f"config:{config}")
    if config != None:
        session_id = config["configurable"]["thread_id"]
    else:
        session_id = None

    docs = search_docs(question, top_k, config)
    # return {"query": question, "docs": docs}

    if session_id is None:
        logger.error("session_id is None in config")
        return {"query": question, "docs": docs}
    # logger.debug(f"检索到的文档{docs}")
    ids = global_reference_map.add_references(session_id, docs)
    if not ids or not docs:
        logger.warning("ids or docs is empty, returning empty result")
        return {"query": question, "docs": []}
    # 先把docs按ids升序排序
    # ["【文档x】name\ncontent\n",...]
    ids, docs = zip(*sorted(zip(ids, docs)))
    # rename_docs = ["【文档" + str(doc_id) + "】" + doc.get("source", "") + "\n" + doc.get("content", "") for doc_id, doc in zip(ids, docs)]
    rename_docs = [
        {
            "source": "【文档" + str(doc_id) + "】" + doc.get("source", ""),
            "content": doc.get("content", ""),
        }
        for doc_id, doc in zip(ids, docs)
    ]
    return {"query": question, "docs": rename_docs}


# todo 知识库的领域分类如何注册到工具调用中？如何根据问题+领域分类，自适应的选择知识库去检索
