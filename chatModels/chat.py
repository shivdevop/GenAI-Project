from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

load_dotenv()

# groq_llm = init_chat_model(
#     "llama-3.1-8b-instant",
#     model_provider="groq"
# )

# response=groq_llm.invoke("what is cricket?")

# print(response.content)

from openai import OpenAI

#create Ai client 
client = OpenAI()

#create conversations 
messages=[
    {
        "role":"system",
        "content":"you are a helpful software professor"
    },
    {
        "role":"user",
        "content":"explain what is an api"
    }
]

#send request to llm 
response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

#extract generated answer 
answer=response.choices[0].message.content

print(answer)