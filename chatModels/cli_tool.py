from dotenv import load_dotenv
from openai import OpenAI
import os 

load_dotenv()

client=OpenAI()

messages=[
    {
        "role":"system",
        "content":"you are a senior backend engineer helping a developer"
    }
]

print("AI CLI CHAT")
print("Type 'exit' to quit\n")


while True:

    #take user input 
    user_input=input("You:")

    if user_input.lower()=="exit":
        break

    messages.append({
        "role":"user",
        "content":user_input
    })

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    ai_response=response.choices[0].message.content

    messages.append({
        "role":"assistant",
        "content":ai_response
    })

    print("\nAI:",ai_response)
    print()

    