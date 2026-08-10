from dotenv import load_dotenv
import os
from groq import Groq



load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
# print(api_key)

client = Groq(api_key=api_key)

response = client.chat.completions.create(model = "llama-3.3-70b-versatile",messages= [
    {"role": "user","content" : "What is the capital of France,and how beautiful it is and so?"}],
    max_completion_tokens=500,
    stream = True
    )

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")