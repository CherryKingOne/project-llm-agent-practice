from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
import pytz

model = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-0e97",
    model="qwen3.5-plus"
)


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间"""
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"当前 {timezone} 时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"获取时间失败: {str(e)}"


from langgraph.graph import StateGraph, END
from typing import TypedDict


class AgentState(TypedDict):
    messages: list
    next_step: str
    script_content: str
    storyboard_content: str
    task_complete: bool


def get_user_message(messages: list) -> str:
    """从 messages 中提取用户消息内容"""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def emma_node(state: AgentState):
    """Emma 调度中心 - 分析任务并调度，接收结果判断是否完成"""
    messages = state["messages"]
    script_content = state.get("script_content", "")
    storyboard_content = state.get("storyboard_content", "")
    task_complete = state.get("task_complete", False)

    if task_complete:
        return {"next_step": "end"}

    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_msg = msg.get("content", "")
            break

    if script_content and not storyboard_content:
        prompt = f"""你叫 Emma，是调度中心 Agent。

编剧已经完成了剧本创作：
{script_content[:500]}...

现在需要判断下一步：
- 剧本已完成，需要调用分镜 Agent 来设计分镜
- 请输出: storyboard_agent"""
    elif script_content and storyboard_content:
        prompt = f"""你叫 Emma，是调度中心 Agent。

当前任务状态：
- 剧本：已完成
- 分镜：已完成

任务已全部完成，请输出: complete"""
    else:
        user_msg = get_user_message(messages)
        prompt = f"""你叫 Emma，是调度中心 Agent。

用户说: {user_msg}

你需要判断应该让哪个员工(Agent)来处理：
- story_writer_agent: 当用户要求写剧本、编剧、创作故事时
- storyboard_agent: 当用户只要求分镜、镜头设计时
- complete: 任务已完成，可以结束

请直接输出要调度的Agent名称（story_writer_agent / storyboard_agent / complete），不要有其他内容。"""

    result = model.invoke(prompt)
    decision = result.content.strip().lower()
    
    if "complete" in decision or (script_content and storyboard_content):
        return {"next_step": "end", "task_complete": True}
    elif "story_writer" in decision:
        return {"next_step": "story_writer"}
    elif "storyboard" in decision:
        return {"next_step": "storyboard"}
    else:
        return {"next_step": "end"}


def story_writer_agent_node(state: AgentState):
    """编剧 Agent 节点"""
    messages = state["messages"]
    user_msg = get_user_message(messages)

    prompt = f"""你是一个专业编剧。请根据以下用户请求创作剧本：

{user_msg}

请输出完整的剧本，包含：
- 故事梗概
- 场景描述
- 角色对话
- 起承转合"""

    result = model.invoke(prompt)
    script_content = result.content
    return {
        "script_content": script_content,
        "messages": state["messages"] + [{"role": "assistant", "content": f"【编剧 Agent 完成剧本创作】\n{script_content[:300]}..."}]
    }


def storyboard_agent_node(state: AgentState):
    """分镜 Agent 节点"""
    messages = state["messages"]
    user_msg = get_user_message(messages)
    script_content = state.get("script_content", "")

    if script_content:
        prompt = f"""你是一个专业分镜师。

编剧已经完成了以下剧本，请根据这个剧本生成分镜脚本：

【剧本内容】
{script_content}

请输出分镜脚本，包含：
- 镜头号
- 景别
- 画面描述
- 镜头运动
- 时长"""
    else:
        prompt = f"""你是一个专业分镜师。请根据以下用户请求生成分镜脚本：

{user_msg}

请输出分镜脚本，包含：
- 镜头号
- 景别
- 画面描述
- 镜头运动
- 时长"""

    result = model.invoke(prompt)
    storyboard_content = result.content
    return {
        "storyboard_content": storyboard_content,
        "messages": state["messages"] + [{"role": "assistant", "content": f"【分镜 Agent 完成分镜设计】\n{storyboard_content[:300]}..."}]
    }


workflow = StateGraph(AgentState)

workflow.add_node("emma", emma_node)
workflow.add_node("story_writer", story_writer_agent_node)
workflow.add_node("storyboard", storyboard_agent_node)

workflow.set_entry_point("emma")

workflow.add_conditional_edges(
    "emma",
    lambda x: x["next_step"],
    {
        "story_writer": "story_writer",
        "storyboard": "storyboard",
        "end": END,
    }
)

workflow.add_edge("story_writer", "emma")
workflow.add_edge("storyboard", "emma")

app = workflow.compile()


def run_conversation():
    print("=" * 50)
    print("多 Agent 调度系统")
    print("工作流: Emma(调度) ↔ 编剧/分镜 Agent")
    print("所有结果返回 Emma 判断是否继续")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("用户: ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit", "q"]:
            print("退出对话")
            break

        print("\n--- Agent 响应 ---")
        
        initial_state = {
            "messages": [{"role": "user", "content": user_input}], 
            "next_step": "", 
            "script_content": "",
            "storyboard_content": "",
            "task_complete": False
        }
        
        try:
            for chunk in app.stream(initial_state):
                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]
                
                if node_name == "emma":
                    next_step = node_output.get("next_step", "unknown")
                    if next_step == "end":
                        print(f"\n[Emma 判断] 任务完成，结束流程")
                    else:
                        print(f"\n[Emma 调度] → 调用 {next_step}")
                else:
                    if "messages" in node_output:
                        last_msg = node_output["messages"][-1]
                        if isinstance(last_msg, dict):
                            print(f"\n{last_msg.get('content', '')}")
        except Exception as e:
            print(f"错误: {e}")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    run_conversation()
