import torch, os, uuid, math, datetime, json
from transformers import MistralForCausalLM, PreTrainedTokenizer
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import Tool, create_json_chat_agent, AgentExecutor
from pathlib import Path
from typing import Optional, List, Any, Type
from collections.abc import Mapping


from langchain.storage import InMemoryByteStore
from langchain.tools import tool

import llmtools
from llmtools import wikipedia_tool, calculator_tool


import logging
logger = logging.getLogger(__name__)

# wrap the LLNM
class CustomLLMMistral(LLM):
    model: MistralForCausalLM
    tokenizer: PreTrainedTokenizer

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:

        messages = [
         {"role": "user", "content": prompt},
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True,
                                            pad_token_id=self.tokenizer.eos_token_id, top_k=4, temperature=0.7)

        decoded = self.tokenizer.batch_decode(generated_ids)

        output = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

        if stop is not None:
          for word in stop:
            output = output.split(word)[0].strip()

        # Mistral 7B sometimes fails to properly close the Markdown Snippets.
        # If they are not correctly closed, Langchain will struggle to parse the output.
        while not output.endswith("```"):
          output += "`"

        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}


llm =   # enter VERTEX AI LLM wrapper here (originally this used: CustomLLMMistral(model=model, tokenizer=tokenizer)  )


system="""
You are an AI agent. You are designed to solve tasks. Each task requires multiple steps that are represented by a markdown code snippet of a JSON blob.

The JSON structure should contain the following keys:
thought -> your current reasoning
action -> name of a tool
action_input -> parameters to send to the tool

These are the tools you can use: {tool_names}.

These are the tools descriptions: {tools}

If you are confident the final answer can be given now (based on internal knowledge or previous tool outputs), use the tool "Final Answer" directly. Its parameter is your direct response. This ends the reasoning.

If there is not enough information, or more tool use is clearly needed, keep reasoning step-by-step.

You must respond with only one step at a time. Do not skip ahead or produce multiple steps in one output.

Do not guess. Base your decisions strictly on the information available from previous steps.
"""

human="""
Add the word "STOP" after each markdown snippet. Format each step like this:

```json
{{"thought": "<your thoughts>",
 "action": "<tool name or Final Answer to give a final answer>",
 "action_input": "<tool parameters or the final output"}}
```
STOP

This is my query="{input}". Write only the next step needed to solve it.
Your answer should be based on the previous tools executions, even if you think you know the answer.
Remember to add STOP after each snippet.

These were the previous steps given to solve this query and the information you already gathered:
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

tools = [wikipedia_tool, calculator_tool]

agent = create_json_chat_agent(
    tools = tools,
    llm = llm,
    prompt = prompt,
    stop_sequence = ["STOP"],
    template_tool_response = "{observation}"
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) 

