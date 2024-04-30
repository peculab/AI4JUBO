from transformers import BitsAndBytesConfig
import torch

tavily_key = "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"
huggingface_key = "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"

'''
	model_id list:
		"epfl-llm/meditron-70b"
		"meta-llama/Meta-Llama-3-70B"
		"taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"

'''
model_id = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
tokenizer_id = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

bits_and_bytes_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 啟用4位元量化
    bnb_4bit_compute_dtype=torch.float16,  # 計算時使用的數據類型
    bnb_4bit_quant_type="nf4",  # 量化類型
    bnb_4bit_use_double_quant=True,  # 使用雙重量化
)

prompt_template = '''
					Answer the following questions as best you can. 
					Answer the details, but don't use too much words, keep the answer short and simple.
					Answer with Chinese instead of English. (if needed)

					You have access to the following tools: 

					{tools}

					Use the following format:

					Question: the input question you must answer

					Thought: you should always think about what to do
					Tool: The name of the tool you use, should be one of [{tool_names}]
					Input: the input to the tool you use
					Observation: the return output of the tool
					...
					(The process Thought/Tool/Input/Observation can be repeated if answer is not confirmed.)
					
					Thought: I now know the final answer
					Final Answer: the final answer to the original input question
					(Strictly follow this format, notice the case and punctuation.)

					After generating all answers, raise a format error if you found out you answer is in a incorrect format or the answer is obviously incorrect.

					Begin!

					Question: {input}
					Thought:{agent_scratchpad}
				'''