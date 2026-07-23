"""
yaml
"""

import yaml
from utils.path_tool import get_abs_path


def load_rag_config(
        config_path: str = get_abs_path("config/rag.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(
        config_path: str = get_abs_path("config/chroma.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



def load_agent_config(
        config_path: str = get_abs_path("config/agent.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(
        config_path: str = get_abs_path("config/prompts.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_cache_config(
        config_path: str = get_abs_path("config/cache.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_db_config(
        config_path: str = get_abs_path("config/db.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_semantic_cache_config(
        config_path: str = get_abs_path("config/semantic_cache.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_celery_config(
        config_path: str = get_abs_path("config/celery.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_document_parser_config(
        config_path: str = get_abs_path("config/document_parser.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_retrieval_config(
        config_path: str = get_abs_path("config/retrieval.yml"),
        encoding: str = "utf-8"
):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf=load_rag_config()
chroma_conf=load_chroma_config()
agent_conf=load_agent_config()
prompts_conf=load_prompts_config()
cache_conf = load_cache_config()
db_conf = load_db_config()
semantic_cache_conf = load_semantic_cache_config()
celery_conf = load_celery_config()
document_parser_conf = load_document_parser_config()
retrieval_conf = load_retrieval_config()