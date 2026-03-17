from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 

load_dotenv()

#define response schema 

#with Field the LLM understands exactly what to generate
class TopicResponse(BaseModel):
    definition: str = Field(description="definition of the topic")
    example: str = Field(description="real-world example ")
    advantages: list[str]
    disadvantages: list[str]

#create parser 
parser= PydanticOutputParser(pydantic_object=TopicResponse)

#model
model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

#prompt 
prompt=PromptTemplate(
    template=""" 
    Explain {topic}.
    
    {format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions":parser.get_format_instructions()
    }
)

#create chain
chain=prompt | model | parser

# invoke llm 

response=chain.invoke({"topic":"Docker"})
print(response)

#this helps print raw Json instead of printing the pydantic object 
print(response.model_dump())