from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.rag.vectorstore import get_vectorstore
from langchain_community.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0.2,
        streaming=True,
        base_url=settings.base_url,
    )
# def get_llm():
#     return ChatOpenAI(
#         model="deepseek-chat",
#         openai_api_key=settings.openai_api_key,
#         openai_api_base="https://api.deepseek.com",
#         temperature=0,
#     )
# def get_embeddings():
#     return OpenAIEmbeddings(
#     api_key=settings.openai_api_key,
#     base_url=settings.base_url,
#     model=settings.model_name)

# def get_embeddings():
#     return OpenAIEmbeddings(
#         model="text-embedding-v4",                  # 阿里云向量模型，推荐新一代
#         api_key=settings.dashscope_api_key,        # 同一个百炼的 key
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )
def get_embeddings():
    # 使用开源嵌入模型，避免额外费用
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # 轻量且效果不错
        model_kwargs={'device': 'cpu'}
    )
def get_vs():
    return get_vectorstore(get_embeddings())


