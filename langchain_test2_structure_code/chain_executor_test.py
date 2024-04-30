import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent

import torch.quantization
from huggingface_hub import login

from tools.math_tools import *
import config

class myAgent():
	def __init__(self):
		os.environ["TAVILY_API_KEY"] = config.tavily_key
		os.environ["HUGGINGFACE_API_KEY"] = config.huggingface_key
		login(os.environ["HUGGINGFACE_API_KEY"])	    
		self.tools = [TavilySearchResults(max_results=5), addTool, subtractTool, divideTool, multiplyTool]

		self.createLLM(config.bits_and_bytes_config, config.model_id, config.tokenizer_id)
		self.createPrompt( config.prompt_template )
		self.createExecutor()

	def createLLM(self, config, model_id, tokenizer_id):
		model_4bit = AutoModelForCausalLM.from_pretrained(
		    model_id,
		    device_map="cuda",  # 自動選擇運行設備
		    quantization_config=config,
		)
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

		text_generation_pipeline = pipeline(
		    "text-generation",
		    model=model_4bit,
		    tokenizer=tokenizer,
		    use_cache=True,
		    device_map="cuda",
		    max_length=100000,
		    do_sample=True,
		    top_k=5,
		    num_return_sequences=1,
		    eos_token_id=tokenizer.eos_token_id,
		    pad_token_id=tokenizer.eos_token_id,
		)

		self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

	def createPrompt(self, promptTemplate, promptId=None):
		self.prompt = PromptTemplate.from_template(promptTemplate)

	def createExecutor(self):
		self.agent = create_react_agent(self.llm, self.tools, self.prompt)
		self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

	def invoke(self, query):
		result = self.executor.invoke({"input": query})
		return result