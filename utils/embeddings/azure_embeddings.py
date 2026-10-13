"""This module provides functions to create instances of Azure OpenAI embeddings and models."""

from langchain_openai import (
    # AzureOpenAIEmbeddings,    # FIXME: Azure OpenAI Embeddings are not available yet.
    ChatOpenAI,
    AzureChatOpenAI,
)
from utils.config import Config
import logging
import os


def __get_openai_embeddings():
    raise NotImplementedError("Azure OpenAI Embeddings are not available yet.")


#     """Create an instance of AzureOpenAIEmbeddings.

#     Returns:
#         AzureOpenAIEmbeddings: An instance of AzureOpenAIEmbeddings.
#     """
#     if not all(
#         [
#             Config.AZURE_OPENAI_API_KEY,
#             Config.AZURE_OPENAI_API_VERSION,
#             Config.AZURE_OPENAI_BASE_URL,
#             Config.AZURE_OPENAI_SUBSCRIPTION_KEY,
#         ]
#     ):
#         logging.warning("Missing Azure OpenAI configuration. Chatbot is not available.")
#         return None

#     try:
#         _emb = AzureOpenAIEmbeddings(
#             api_key="<REDACTED>",
#             azure_endpoint="<REDACTED>",
#             api_version="2024-12-01-preview",
#             model="<REDACTED>",
#         )
#     except Exception as e:
#         logging.error(f"Error creating OpenAI embeddings: {e}")
#         _emb = None

# return _emb


def __get_openai_model(
    model: str = str(Config.AZURE_OPENAI_CHAT_MODEL),
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 30,
    max_retries: int = 2,
) -> AzureChatOpenAI | None:
    """Create an instance of AzureChatOpenAI.

    Args:
        model (str): The model name to use.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens to generate.
        timeout (int): Timeout for the request.
        max_retries (int): Maximum number of retries for the request.
    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI.
    """
    try:

        # _model: AzureChatOpenAI | None = AzureChatOpenAI(
        #     api_key=Config.AZURE_OPENAI_API_KEY,
        #     azure_endpoint=Config.AZURE_OPENAI_BASE_URL,
        #     api_version=Config.AZURE_OPENAI_API_VERSION,
        #     model=model,
        #     max_tokens=max_tokens,
        #     max_retries=max_retries,
        #     timeout=timeout,
        #     temperature=temperature,
        # )
        _model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
    except Exception as e:
        logging.error(f"Error creating OpenAI model: {e}")
        _model = None

    return _model
