import openai
from openai import OpenAI

# Use your real API key here
client =  OpenAI(api_key="sk-XXXXXXX")

print("Talk to Buddy! (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly chatbot named Buddy."},
            {"role": "user", "content": user_input}
        ]
    )

    print("Buddy:", response.choices[0].message.content)
