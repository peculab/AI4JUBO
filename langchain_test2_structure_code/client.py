from chain_executor_test import myAgent
import config

def main():
	agent = myAgent()
	query = "What is the population of Taiwan?"
	result = agent.invoke(query)
	print(f"Input: {query} \n Output:{result['output']}")

if __name__ == '__main__':
	main()