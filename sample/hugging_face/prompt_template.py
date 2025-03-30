from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    do_sample=True,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)

# Create a LangChain LLM from the pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Example with prompt template.
prompt = PromptTemplate.from_template("""
Question: {question}

Answer: Let's think step by step.
""")

chain = prompt | llm

response = chain.invoke({"question": "What is the largest ocean in the world?"})
print(response)