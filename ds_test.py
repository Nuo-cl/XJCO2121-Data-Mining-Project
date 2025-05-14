from openai import OpenAI

client = OpenAI(api_key="sk-9da535bc9b544bdda2418983dc6e3666",
                base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)