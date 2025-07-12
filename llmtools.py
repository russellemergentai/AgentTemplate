import torch, os, uuid, math, datetime, json
import numexpr as ne
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.language_models.llms import LLM
from langchain.agents import Tool
from pathlib import Path
from typing import Optional, List, Any, Type
from collections.abc import Mapping
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.tools import tool

import commonllm
from commonllm import model, tokenizer

import logging
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = ""

########WIKI TOOL########
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2500))

wikipedia_tool = Tool(
    name="wikipedia",
    description="Use this tool for wiki searches. Never search for more than one concept at a single step. If you need to compare two concepts, search for each one individually. Syntax: string with a simple concept",
    func=wikipedia.run
)


########CALCULATOR TOOL########
class Calculator(BaseTool):
    name: str = "calculator"
    description: str = (
        "Use this tool for math operations only. It requires numexpr syntax. "
        "Use it always if you need to solve any arithmetic operation. Be sure syntax is correct."
    )

    def _run(self, expression: str):
        try:
            return ne.evaluate(expression).item()
        except Exception as e:
            return f"This is not a numexpr valid syntax. Error: {str(e)}"

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

calculator_tool = Calculator()

