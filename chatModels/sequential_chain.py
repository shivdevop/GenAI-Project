from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

load_dotenv()
model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

#FIRST FETCH TOPIC FROM USER INPUT 

class Topic(BaseModel):
    topic: str 


topic_parser=PydanticOutputParser(pydantic_object=Topic) 
#response should be having this schema 


topic_prompt=PromptTemplate(
   template= """
Extract main topic from user input
Input:{input}

{format_instructions}
""",
input_variables=["input"],
partial_variables={"format_instructions":topic_parser.get_format_instructions()}
)

topic_chain= topic_prompt | model | topic_parser


#EXPLAIN TOPIC EXTRACTED FROM USER INPUT 

class Description(BaseModel):
    description: str

description_parser=PydanticOutputParser(pydantic_object=Description)

description_prompt=PromptTemplate(
    template="""
Explain the topic clearly

{format_instructions}

Topic:{topic}

""",
input_variables=["topic"],
partial_variables={"format_instructions":description_parser.get_format_instructions()}
)

description_chain=description_prompt | model | description_parser

#GENERATE EXAMPLE BASED ON THIS DESCRIPTION 

class Example(BaseModel):
    example: str 

example_parser=PydanticOutputParser(pydantic_object=Example)

example_prompt=PromptTemplate(
    template="""
Give a real use case example based on the description

{format_instructions}

Description:{description}
""",
input_variables=["description"],
partial_variables={"format_instructions":example_parser.get_format_instructions()}
)

example_chain=example_prompt | model | example_parser

#RUN THE PIPELINE 

user_input="Can you explain Docker container"

topic=topic_chain.invoke({"input":user_input})
explanation=description_chain.invoke({"topic":topic.topic})
example=example_chain.invoke({"description":explanation.description})

print("\nTOPIC:", topic.topic)
print("\nDEFINITION:", explanation.description)
print("\nEXAMPLE:", example.example)
