import subprocess
import re
from openai import OpenAI

client = OpenAI(
    api_key="234695734ae84aa692b641bca81eb28b.kSdKlEgDpmATAoh8",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 系统提示词，定义 Agent 身份和工具使用权限
system_prompt = """you are Agent you name is Diyclaw

你有权限使用以下工具：
1. command - 用于执行系统命令和程序，格式: (command) <命令内容>
2. text - 用于生成和编辑文本内容，格式: (text) <文本内容>

当用户请求需要执行操作时，请使用上述格式声明你要使用的工具。
如果命令执行失败，请分析错误原因并尝试其他方案继续完成任务。"""

# 历史消息列表
history_messages = [
    {"role": "developer", "content": system_prompt}
]


def execute_command(command):
    """执行系统命令并返回结果"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip() or "命令执行完成（无输出）"}
        else:
            return {"success": False, "output": result.stderr.strip() or "命令执行失败"}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "命令执行超时"}
    except Exception as e:
        return {"success": False, "output": f"执行出错: {str(e)}"}


def extract_tool_calls(response):
    """从 AI 回复中提取所有工具调用"""
    tools = []

    # 匹配 (command) 命令格式
    for match in re.finditer(r'\(command\)\s*(.+?)(?=\n\(command\)|\n\(text\)|\Z)', response, re.IGNORECASE | re.DOTALL):
        command = match.group(1).strip()
        tools.append(("command", command))

    # 匹配 (text) 文本格式
    for match in re.finditer(r'\(text\)\s*(.+?)(?=\n\(command\)|\n\(text\)|\Z)', response, re.IGNORECASE | re.DOTALL):
        text_content = match.group(1).strip()
        tools.append(("text", text_content))

    return tools


def process_ai_response(response):
    """解析 AI 回复，执行其中的工具调用"""
    tools = extract_tool_calls(response)

    if not tools:
        return None, None

    results = []
    for tool_type, tool_content in tools:
        if tool_type == "command":
            print(f"[执行命令] {tool_content}")
            result = execute_command(tool_content)
            status = "成功" if result["success"] else "失败"
            print(f"[执行{status}] {result['output'][:500]}")
            results.append(f"命令 '{tool_content}' 执行{status}: {result['output']}")
        elif tool_type == "text":
            print(f"[生成文本] {tool_content[:100]}...")
            results.append(f"生成文本: {tool_content}")

    return "\n".join(results), bool(tools)


def get_ai_response(messages):
    """获取 AI 回复"""
    completion = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=messages,
    )
    return completion.choices[0].message.content


print("=== Diyclaw 对话助手 ===")
print("输入 'exit' 或 'quit' 退出对话\n")

while True:
    # 获取用户输入
    user_message = input("你: ")

    # 检查是否退出
    if user_message.lower() in ['exit', 'quit', '退出']:
        print("再见！")
        break

    # 将用户消息添加到历史
    history_messages.append({"role": "user", "content": user_message})

    # 循环执行：AI 思考 -> 执行工具 -> 反馈结果 -> AI 继续处理
    max_iterations = 10  # 防止无限循环
    for iteration in range(max_iterations):
        # 获取 AI 回复
        ai_response = get_ai_response(history_messages)

        if iteration == 0:
            print(f"Diyclaw: {ai_response}\n")
        else:
            print(f"[AI 继续处理] {ai_response[:200]}...\n")

        # 执行 AI 回复中的工具调用
        tool_result, has_tools = process_ai_response(ai_response)

        # 添加 AI 回复到历史
        history_messages.append({"role": "assistant", "content": ai_response})

        if not has_tools:
            # AI 没有调用工具，任务完成
            break

        # 将工具执行结果反馈给 AI，让它继续处理
        if tool_result:
            history_messages.append({"role": "user", "content": f"[工具执行结果] {tool_result}"})
    else:
        print("[达到最大迭代次数，停止执行]")
