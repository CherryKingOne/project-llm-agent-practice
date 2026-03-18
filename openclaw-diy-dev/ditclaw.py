from openai import OpenAI

client = OpenAI(
    api_key="234695734ae84aa692b641bca81eb28b.kSdKlEgDpmATAoh8",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

completion = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "developer", "content": "you are Agent you name is Diyclaw"},
        {
            "role": "user",
            "content": "你是谁",
        },
    ],
)

print(completion.choices[0].message.content)