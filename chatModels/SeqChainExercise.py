"""
Objective:
Take user input and build a pipeline where:
Step 1 → Extract topic
Step 2 → Generate advantages & disadvantages
Step 3 → Summarize pros/cons
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Topic prompt and parser 
class Topic(BaseModel):
    topic: str 
    
topic_parser=PydanticOutputParser(pydantic_object=Topic)
topic_prompt=PromptTemplate(
    template="""
Extract main topic from the user input
{format_instructions}
Input:{input}
""",
input_variables=["input"],
partial_variables={"format_instructions":topic_parser.get_format_instructions()}
)    

topic_chain= topic_prompt | model | topic_parser

# Advantage and disadvantage prompt and parser 

class Analysis(BaseModel):
    advantages: list[str] = Field(description="list of advantages")
    disadvantages: list[str] = Field(description="list of disadvantages")

analysis_parser=PydanticOutputParser(pydantic_object=Analysis)

analysis_prompt=PromptTemplate(
    template="""
Give a list of advantages and disadvantages based on the topic
{format_instructions}
Topic:{topic}
""",
input_variables=["topic"],
partial_variables={"format_instructions":analysis_parser.get_format_instructions()}
)

analysis_chain=analysis_prompt | model | analysis_parser

# summarize prompt and parser 

class Summary(BaseModel):
    summary: str= Field(description="short summary of pros and cons")

summary_parser=PydanticOutputParser(pydantic_object=Summary)

summary_prompt=PromptTemplate(
    template="""
Give a summary of the advantages and disadvantages
{format_instructions}
Advantages:{advantages}
Disadvantages:{disadvantages}""",
input_variables=["advantages","disadvantages"],
partial_variables={"format_instructions":summary_parser.get_format_instructions()}
)

summary_chain=summary_prompt | model | summary_parser


#run the pipeline
user_input="Explain what is docker"

topic=topic_chain.invoke({"input":user_input})

analysis=analysis_chain.invoke({
    "topic":topic.topic
})

summary=summary_chain.invoke({
    "advantages":analysis.advantages,
    "disadvantages":analysis.disadvantages
})

# -------------------------
# Output
# -------------------------

print("\nTOPIC:", topic.topic)

print("\nADVANTAGES:")
for adv in analysis.advantages:
    print("-", adv)

print("\nDISADVANTAGES:")
for dis in analysis.disadvantages:
    print("-", dis)

print("\nSUMMARY:", summary.summary)