# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationSummaryMemory
# from langchain.chains import ConversationChain 

# model=ChatOpenAI(
#     model="chat-4o-mini",
#     temperature=0.2
# )

# memory=ConversationSummaryMemory(llm=model)

# conversation=ConversationChain(
#     llm=model,
#     memory=memory,
#     verbose=True
# )

# while True:
#     user_input="You : \n"
#     response= conversation.predict(input=user_input)
#     print("AI: \n")





"""
1)We are going to build summary memory system manually by building prompts ourselves 
2) The system will use 2 llm calls per message -> one for response, one for summary update 
3) with the framework we are missing out on controlling prompt building here and memory injection in the pipeline 

"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

SYSTEM_PROMPT="You are a helpful AI Assistant"

print("AI chatbot with summary memory. Type 'exit' to quit.\n ")

summary=""

#using chat_history[] we will only track the convo between user and the llm model and we will keep truncating it.
#we will provide this limited chat history also along with conversation summary to enhance the memory architecture.
chat_history=[]


while True:

    user_input=input("You: \n")
    
    if user_input.lower()=="exit":
        break

    # Build Messages 
    messages=[SystemMessage(content=SYSTEM_PROMPT),
              SystemMessage(content=f"Summary of conversation so far:{summary}")] + chat_history

    messages.append(HumanMessage(content=user_input))

    # Invoke LLM and print the response 

    ai_response=model.invoke(messages)
    print("AI: \n",ai_response.content)

    # Update chat memory with recent human and ai interaction 
    
    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ai_response.content))

    #limit recent messages 
    if len(chat_history)>4:
        chat_history=chat_history[-4:]

    # Update the summary with the latest response (so prompt ai and get a new summary for fresh context after this latest interaction b/w user and llm )

    summary_prompt=[SystemMessage(content="Summarize the content briefly"),
                    HumanMessage(content=f"""
Current Summary:{summary}

New chat:
User:{user_input}
Ai: {ai_response.content}

Update the summary:
""")]
    
    summary_response=model.invoke(summary_prompt)
    summary=summary_response.content

    print(f"Updated Summary: {summary}\n")
    print("---------------------\n")

