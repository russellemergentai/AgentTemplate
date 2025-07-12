import torch, os, uuid, math, datetime, json
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from typing import Optional, List, Any, Type
from collections.abc import Mapping
from langchain.storage import InMemoryByteStore
from langchain.tools import tool
import llmtools, mistralllm, cottools, commonllm
from mistralllm import agent_executor, memory, agent, tools
 
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


from flask import Flask, request
from flask_cors import CORS
import os
import torch
from langchain.agents import AgentExecutor


app = Flask(__name__)
CORS(app)  # This enables CORS for all routes and origins


# Initial agent setup
clean_cuda()   

prompt = ""

@app.route('/')
def home():
    global prompt
    
    try:
        query = request.args.get('query', '').strip()
    
        if not query:
            return "No query provided."

        prompt = query
    
        response = agent_executor.invoke(
            {"input": query}}
        )

        res = response["output"]
        
        return f"RESULT: '{res}'\nfrom PROMPT: '{prompt}'"

    except Exception as e:
        logger.error("Error during request handling:", e)
        return f"Error: {e}"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)