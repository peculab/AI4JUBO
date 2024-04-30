
from langchain.chains import LLMMathChain
from langchain.agents import Tool

from langchain_community.llms import CTransformers
llm = CTransformers(model='TheBloke/Llama-2-7B-GGUF', model_file="llama-2-7b.Q5_K_M.gguf", model_type="llama", gpu_layers=50)
#llm = CTransformers(model='QuantFactory/Meta-Llama-3-8B-GGUF', model_file="Meta-Llama-3-8B.Q4_K_M.gguf", model_type="llama", gpu_layers=50)

import requests

from langchain.tools import BaseTool

class check_site_alive(BaseTool):
    
     name = "check a site is alive"
     description = "Check a site is alive or not."
     
     def _run(self, site: str):
        try:
            resp = requests.get(f'https://{site}')
            resp.raise_for_status()
            print(site,"YesQQ")
            return True
        except Exception:
            print(site,"NoQQ")
            return False
     def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

llm_math = LLMMathChain(llm = llm)
# initialize the math tool
math_tool = Tool(
    name ='Calculator',
    func = llm_math.run,
    description ='Useful for when you need to answer questions about math.'
)
# when giving tools to LLM, we must pass as list of tools
tools = [math_tool, check_site_alive()]

from langchain.agents import initialize_agent
zero_shot_agent = initialize_agent(
    agent = "zero-shot-react-description",
    tools = tools,
    llm = llm,
    verbose = True,
    max_iterations = 3
)

#r = zero_shot_agent(" what is (4.5*2.1)^2.2?")
r = zero_shot_agent.invoke(" what is 1+1=?")
print(r)


r = zero_shot_agent.invoke("example.com site is alive?")
print(r)

#r = zero_shot_agent.invoke("e4xaddmple.com site is alive?")
#print(r)

#r = zero_shot_agent.invoke("www.google.com site is alive?")
#print(r)
