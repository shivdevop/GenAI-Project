"""
Create a memory store 
keep passing the previous conversation history along with the new query 

--> ALSO TRY AND LIMIT HISTORY TO AVOID EXCEEDING THE CONTEXT WINDOW LIMIT !!!!
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

#memory store 
chat_memory=[]

print("AI Chat with Memory (type 'exit' to quit)\n")

while True:

    user_input=input("You: ")

    if user_input.lower()=="exit":
        break

    #Add human message 
    chat_memory.append(HumanMessage(content=user_input))

    #call llm with full history 
    response=model.invoke(chat_memory)

    #Add ai response to chat memory 
    chat_memory.append(AIMessage(content=response.content))

    print("\nAI: ", response.content, "\n")


