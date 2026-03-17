from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

#load env variables 
load_dotenv()

#create model 
model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

#create prompt using template 
prompt=PromptTemplate(
    template="Explain {topic} in simple terms in 50 words max",
    input_variables=["topic"]
)

#create chain 
chain=prompt | model
#LCEL (LangChain Expression Language).

#invoke chain for response 
response=chain.invoke({"topic":"Docker"})

#print response 
print(response.content)