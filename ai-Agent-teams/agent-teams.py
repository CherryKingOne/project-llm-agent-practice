from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
import pytz

model = ChatOpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key="ms-2ecf5308-d03f-4f7c-b4c1-2fc7e789045f",
    model="ZhipuAI/GLM-5"
)

response = model.invoke("你是谁")
print(response.content)

