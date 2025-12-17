"""
AI Agent 自动化测试框架 - LangGraph 执行图
"""
import os
import sys
import asyncio
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from typing_extensions import Optional
from action.enhance_task import EnhanceTaskAction
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from business_knowledge.database import get_db as get_business_db
from reasoning_knowledge.database import get_db as get_reasoning_db
from task_storage.database import get_db as get_task_db
from business_knowledge.crud import BusinessKnowledgeCRUD
import config
from action.decompose_task import DecomposeTaskAction
from action.ui_tars import UITars
from task_storage.database import get_db as get_task_db
from task_storage.crud import TaskStorageCRUD
import json
from action.judgment_task import JudgmentTask
from reasoning_knowledge.crud import ReasoningKnowledgeCRUD

# 配置（从全局配置字典获取）
OPENAI_API_KEY = config.config_dict.get("OPENAI_API_KEY", "sk-f9b24b3890b044ff9d2dfe55d527c6c3")
OPENAI_BASE_URL = config.config_dict.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = config.config_dict.get("MODEL_NAME", "qwen-vl-max")

UI_TARS_BASE_URL = config.config_dict.get("UI_TARS_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/")
UI_TARS_API_KEY = config.config_dict.get("UI_TARS_API_KEY", "dd4d8525-2aec-4159-b693-e7451131a795")
UI_TARS_MODEL = config.config_dict.get("UI_TARS_MODEL", "doubao-1-5-ui-tars-250428")

# 初始化 LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
    temperature=0.3,
)

tars = UITars(base_url=UI_TARS_BASE_URL, api_key=UI_TARS_API_KEY, model=UI_TARS_MODEL)

# === 状态定义 ===
class AgentState(TypedDict):
    """Agent 执行状态"""
    # 输入
    original_task: str  # 原始任务描述
    
    # 补全后的任务
    enhanced_task: Optional[str]  # 补全后的任务描述
    
    # 执行判断
    can_execute: Optional[bool]  # 是否能够执行
    execution_reason: Optional[str]  # 不能执行的原因
    
    # 子任务
    steps: List[Dict[str, Any]]  # 拆解后的步骤列表
    current_step_index: int  # 当前执行的步骤索引
    
    # 执行结果
    step_results: List[Dict[str, Any]]  # 步骤执行结果
    final_result: Optional[Dict[str, Any]]  # 最终结果
    
    # 消息历史
    messages: Annotated[List, add_messages]  # 消息历史


# === 节点函数 ===

async def enhance_task_node(state: AgentState) -> AgentState:
    """
    节点1: 补全执行任务
    使用 LLM 补全和优化任务描述
    """
    print(f"[补全任务节点] 原始任务: {state['original_task']}")
    db = next(get_business_db())
    crud = BusinessKnowledgeCRUD(db)
    # 根据问题搜索
    results = crud.search_by_question(
        query_text=state['original_task'],
        top_k=5,
        threshold=0.5
    )
    background_knowledge = "\n".join([result['answer_text'] for result in results])

    action = EnhanceTaskAction(llm)
    enhanced_task = await action.run(background_knowledge=background_knowledge, original_task=state['original_task'])
    
    print(f"[补全任务节点] 补全后任务: {enhanced_task}")

    return {
        **state,
        "enhanced_task": enhanced_task,
        "messages": [AIMessage(content=f"补全后的任务: {enhanced_task}")]
    }


async def check_executability_node(state: AgentState) -> AgentState:
    """
    节点2: 判断是否能够执行
    使用 LLM 判断任务是否可执行
    """
    print(f"[判断可执行性节点] 检查任务: {state['enhanced_task']}")

    #历史任务知识库
    db = next(get_task_db())
    crud = TaskStorageCRUD(db)
    results = crud.search_by_enhanced_task(
        query_text=state['enhanced_task'],
        top_k=3,
        threshold=0.8
    )
    if results:
        history_tasks = "\n".join([result['enhanced_task']  + ": 可以执行" if result['all_success'] else result['enhanced_task']  + ": 不能执行" + result['execution_reason']  for result in results])
    else:
        history_tasks = ""

    # 业务知识知识库
    db_BusinessKnowledge = next(get_business_db())
    crud_BusinessKnowledge = BusinessKnowledgeCRUD(db_BusinessKnowledge)

    results = crud_BusinessKnowledge.search_by_question(
        query_text=state['enhanced_task'],
        top_k=5,
        threshold=0.5
    )
    if results:
        background_knowledge = "\n".join([result['answer_text'] for result in results])
    else:
        background_knowledge = ""

    # 推理知识库
    #db_reason = next(get_reasoning_db())
    #crud_reason = ReasoningKnowledgeCRUD(db_reason)
    #results = crud.search_by_task(
        #query_text=state['enhanced_task'],
        #top_k=2,
        #threshold=0.8
    #)
    #if results:
        #reason_tasks = "\n\n".join([result['task_text'] + "\n" + result['step_text'] for result in results])
    #else:
        #reason_tasks = ""
    
    action = JudgmentTask(llm)
    can_execute, execution_reason = await action.run(history_tasks=history_tasks, background_knowledge=background_knowledge,task=state['enhanced_task'])

    print(f"[判断可执行性节点] 可执行: {can_execute}, 原因: {execution_reason}")
    
    return {
        **state,
        "can_execute": can_execute,
        "execution_reason": execution_reason,
        "messages": [AIMessage(content=f"可执行性判断: {'可执行' if can_execute else '不可执行'} - {execution_reason}")]
    }


async def decompose_task_node(state: AgentState) -> AgentState:
    """
    节点3: 拆解子任务
    将任务拆解成多个可执行的步骤
    """
    print(f"[拆解任务节点] 拆解任务: {state['enhanced_task']}")

    db = next(get_reasoning_db())
    crud = ReasoningKnowledgeCRUD(db)
    results = crud.search_by_task(
        query_text=state['enhanced_task'],
        top_k=2,
        threshold=0.8
    )
    if results:
        history_tasks = "\n\n".join([result['task_text'] + "\n" + result['step_text'] for result in results])
    else:
        history_tasks = ""

    action = DecomposeTaskAction(llm)
    steps = await action.run(history_tasks=history_tasks, task=state['enhanced_task'])
    print(f"[拆解任务节点] 拆解出 {len(steps)} 个步骤: {steps}")
    
    return {
        **state,
        "steps": steps,
        "current_step_index": 0,
        "step_results": [],
        "messages": [AIMessage(content=f"拆解出 {len(steps)} 个步骤")]
    }


async def execute_subtask_node(state: AgentState) -> AgentState:
    """
    节点4: 执行子任务
    """
    current_step_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    step_results = state.get("step_results", [])
    
    if current_step_index >= len(steps):
        print(f"[执行子任务节点] 所有子任务已完成")
        return state
    
    current_step = steps[current_step_index]
    print(f"[执行子任务节点] 执行子任务 {current_step_index + 1}/{len(steps)}: {current_step.get('step', '')}")
    
    try:
        instruction = """
        你需要执行‘现在需要执行的步骤’中的步骤。

        # 总任务
        {task}

        # 先前已经执行的步骤
        {history_steps}

        # 现在需要执行的步骤
        {need_execute_step}

        # 注意事项
        可根据实际情况调整‘现在需要执行的步骤’中的步骤，但要保证调整后的步骤所做的事与‘现在需要执行的步骤’一致。
        """

        task = state.get("enhanced_task", "")
        history_steps = "\n".join([f"{idx+1}. {step['step_description']}" for idx, step in enumerate(step_results)])
        need_execute_step = current_step['step']

        result = await tars.run(instruction=instruction.format(task=task, history_steps=history_steps, need_execute_step=need_execute_step))
        print(result)
        print(f"[执行子任务节点] 执行结果: {result.get('success', False)}")

        # 将结果转换为 step_result 格式
        step_result = {
            "step_id": current_step.get("id"),
            "step_description": current_step.get("step"),
            "success": result.get("success", False) if isinstance(result, dict) else False,
        }
        
        step_results.append(step_result)
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行完成")
                
    except Exception as e:
        import traceback
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行失败: {str(e)}")
        print(f"[执行子任务节点] 错误详情: {traceback.format_exc()}")
        step_result = {
            "step_id": current_step.get("id"),
            "step_description": current_step.get("step"),
            "success": False,
            "error": str(e),
        }
        step_results.append(step_result)
    
    # 移动到下一个子任务
    next_index = current_step_index + 1
    
    return {
        **state,
        "current_step_index": next_index,
        "step_results": step_results,
        "messages": [AIMessage(content=f"子任务 {current_step_index + 1} 执行完成")]
    }


async def finalize_node(state: AgentState) -> AgentState:
    """
    节点5: 结束节点
    汇总所有执行结果
    """
    print(f"[结束节点] 汇总执行结果")
    
    step_results = state.get("step_results", [])
    all_success = bool(step_results) and all(r.get("success", False) for r in step_results)
    
    final_result = {
        "original_task": state.get("original_task"),
        "enhanced_task": state.get("enhanced_task"),
        "total_steps": len(state.get("steps", [])),
        "completed_steps": len(step_results),
        "all_success": all_success,
        "step_results": step_results,
        "summary": f"共执行 {len(step_results)} 个步骤，{'全部成功' if all_success else '部分失败'}"
    }
    
    print(f"[结束节点] 最终结果: {final_result['summary']}")
    
    return {
        **state,
        "final_result": final_result,
        "messages": [AIMessage(content=f"任务执行完成: {final_result['summary']}")]
    }


# === 条件边函数 ===

def should_continue_execution(state: AgentState) -> Literal["decompose", "end"]:
    """
    判断是否继续执行：如果可执行则拆解任务，否则结束
    """
    can_execute = state.get("can_execute", False)
    if can_execute:
        return "decompose"
    else:
        return "end"


def should_continue_subtasks(state: AgentState) -> Literal["execute_subtask", "finalize"]:
    """
    判断是否继续执行子任务：如果还有未执行的子任务则继续，否则结束
    """
    current_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    
    if current_index < len(steps):
        return "execute_subtask"
    else:
        return "finalize"


# === 构建执行图 ===

def create_workflow_graph() -> StateGraph:
    """创建 LangGraph 执行图"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("enhance_task", enhance_task_node)
    workflow.add_node("check_executability", check_executability_node)
    workflow.add_node("decompose_task", decompose_task_node)
    workflow.add_node("execute_subtask", execute_subtask_node)
    workflow.add_node("finalize", finalize_node)
    
    # 设置入口点
    workflow.set_entry_point("enhance_task")
    
    # 添加边
    workflow.add_edge("enhance_task", "check_executability")
    workflow.add_conditional_edges(
        "check_executability",
        should_continue_execution,
        {
            "decompose": "decompose_task",
            "end": "decompose_task"
        }
    )
    workflow.add_edge("decompose_task", "execute_subtask")
    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue_subtasks,
        {
            "execute_subtask": "execute_subtask",  # 循环执行子任务
            "finalize": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# === 主函数 ===

async def run_task(task: str, task_id: Optional[int] = None) -> Dict[str, Any]:
    """
    执行一个任务
    
    Args:
        task: 任务描述
        task_id: 任务 ID（如果已存在记录，则更新；否则创建新记录）
        
    Returns:
        执行结果，包含 task_id
    """

    # 创建执行图
    app = create_workflow_graph()
    
    # 初始化状态
    initial_state: AgentState = {
        "original_task": task,
        "enhanced_task": None,
        "can_execute": None,
        "execution_reason": None,
        "steps": [],
        "current_step_index": 0,
        "step_results": [],
        "final_result": None,
        "messages": [],
    }
    
    # 执行图
    print(f"\n{'='*60}")
    print(f"开始执行任务: {task}")
    print(f"{'='*60}\n")
    
    
    final_state = await app.ainvoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"任务执行完成")
    print(f"{'='*60}\n")
    
    # 获取最终结果
    final_result = final_state.get("final_result", {})
    
    # 保存到数据库
    db = next(get_task_db())
    crud = TaskStorageCRUD(db)
    
    try:
        # 准备数据
        steps_str = json.dumps(final_state.get("steps", []), ensure_ascii=False, indent=2) if final_state.get("steps") else None
        step_results_json = final_state.get("step_results", [])
        final_result_str = final_result.get("summary", "")
        
        if task_id:
            # 更新现有记录
            updated_task = crud.update(
                task_id=task_id,
                original_task=final_state.get("original_task"),
                enhanced_task=final_state.get("enhanced_task"),
                can_execute=final_state.get("can_execute"),
                execution_reason=final_state.get("execution_reason"),
                steps=steps_str,
                step_results=step_results_json,
                final_result=final_result_str,
                all_success=final_result.get("all_success", False)
            )
            if updated_task:
                final_result["task_id"] = updated_task.id
        else:
            # 创建新记录
            new_task = crud.create(
                original_task=final_state.get("original_task"),
                enhanced_task=final_state.get("enhanced_task"),
                can_execute=final_state.get("can_execute"),
                execution_reason=final_state.get("execution_reason"),
                steps=steps_str,
                step_results=step_results_json,
                final_result=final_result_str,
                all_success=final_result.get("all_success", False)
            )
            final_result["task_id"] = new_task.id
            print(f"[数据库] 任务已保存，ID: {new_task.id}")
    except Exception as e:
        print(f"[数据库] 保存任务失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    return final_result


async def main():
    """主函数"""
    # 示例任务
    example_task = "测试地铁里视频会议的真实体验"
    
    result = await run_task(example_task)
    
    print("\n执行结果:")
    print(f"原始任务: {result.get('original_task')}")
    print(f"补全任务: {result.get('enhanced_task')}")
    print(f"子任务数: {result.get('total_subtasks')}")
    print(f"完成数: {result.get('completed_subtasks')}")
    print(f"全部成功: {result.get('all_success')}")
    print(f"摘要: {result.get('summary')}")


if __name__ == "__main__":
    asyncio.run(main())

