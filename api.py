"""
FastAPI 接口 - 对外提供启动任务的入口
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from main import run_task
from task_storage.database import get_db as get_task_db, init_db
from task_storage.crud import TaskStorageCRUD
from business_knowledge.database import get_db as get_business_db, init_db as init_business_db
from business_knowledge.crud import BusinessKnowledgeCRUD
from reasoning_knowledge.database import get_db as get_reasoning_db, init_db as init_reasoning_db
from reasoning_knowledge.crud import ReasoningKnowledgeCRUD

# 创建 FastAPI 应用
app = FastAPI(title="AI 任务执行 API", version="1.0.0")


# 请求模型
class TaskRequest(BaseModel):
    """任务请求模型"""
    task: str
    task_id: Optional[int] = None  # 如果提供，则更新现有任务


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: int
    message: str
    status: str


# 业务知识库请求/响应模型
class BusinessKnowledgeCreate(BaseModel):
    """业务知识库创建请求模型"""
    question_text: str
    answer_text: str


class BusinessKnowledgeUpdate(BaseModel):
    """业务知识库更新请求模型"""
    question_text: Optional[str] = None
    answer_text: Optional[str] = None


class BusinessKnowledgeSearch(BaseModel):
    """业务知识库搜索请求模型"""
    query_text: str
    top_k: int = 5
    threshold: float = 0.0


# 推理知识库请求/响应模型
class ReasoningKnowledgeCreate(BaseModel):
    """推理知识库创建请求模型"""
    task_text: str
    step_text: str


class ReasoningKnowledgeUpdate(BaseModel):
    """推理知识库更新请求模型"""
    task_text: Optional[str] = None
    step_text: Optional[str] = None


class ReasoningKnowledgeSearch(BaseModel):
    """推理知识库搜索请求模型"""
    query_text: str
    top_k: int = 5
    threshold: float = 0.0


# 任务存储搜索请求模型
class TaskSearch(BaseModel):
    """任务存储搜索请求模型"""
    query_text: str
    top_k: int = 5
    threshold: float = 0.0


# 初始化数据库
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    try:
        init_db()  # 任务存储数据库
        init_business_db()  # 业务知识库
        init_reasoning_db()  # 推理知识库
        print("[API] 所有数据库初始化完成")
    except Exception as e:
        print(f"[API] 数据库初始化失败: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI 任务执行 API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/api/tasks/run", response_model=TaskResponse)
async def run_task_api(request: TaskRequest):
    """
    启动任务执行（异步）
    
    Args:
        request: 任务请求
        
    Returns:
        任务响应，包含 task_id
    """
    if not request.task or not request.task.strip():
        raise HTTPException(status_code=400, detail="任务描述不能为空")
    
    # 如果提供了 task_id，先检查任务是否存在
    if request.task_id:
        db = next(get_task_db())
        try:
            crud = TaskStorageCRUD(db)
            existing_task = crud.get_by_id(request.task_id)
            if not existing_task:
                raise HTTPException(status_code=404, detail=f"任务 ID {request.task_id} 不存在")
        finally:
            db.close()
    
    # 在后台执行任务（使用 asyncio.create_task）
    async def execute_task():
        """后台执行任务"""
        try:
            result = await run_task(request.task, request.task_id)
            print(f"[API] 任务执行完成，ID: {result.get('task_id')}")
        except Exception as e:
            print(f"[API] 任务执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 使用 asyncio.create_task 在后台执行
    asyncio.create_task(execute_task())
    
    # 如果是更新任务，返回现有 task_id
    if request.task_id:
        return TaskResponse(
            task_id=request.task_id,
            message="任务已添加到执行队列（更新模式）",
            status="queued"
        )
    
    # 如果是新任务，先创建一条记录
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        new_task = crud.create(original_task=request.task)
        task_id = new_task.id
    finally:
        db.close()
    
    # 更新任务 ID 并执行（使用 asyncio.create_task）
    async def execute_task_with_id():
        """使用创建的任务 ID 执行任务"""
        try:
            result = await run_task(request.task, task_id)
            print(f"[API] 任务执行完成，ID: {result.get('task_id')}")
        except Exception as e:
            print(f"[API] 任务执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 使用 asyncio.create_task 在后台执行
    asyncio.create_task(execute_task_with_id())
    
    return TaskResponse(
        task_id=task_id,
        message="任务已添加到执行队列",
        status="queued"
    )


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: int):
    """
    获取任务详情
    
    Args:
        task_id: 任务 ID
        
    Returns:
        任务详情
    """
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        task = crud.get_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 ID {task_id} 不存在")
        return task.to_dict()
    finally:
        db.close()


@app.get("/api/tasks")
async def list_tasks(skip: int = 0, limit: int = 100):
    """
    获取任务列表
    
    Args:
        skip: 跳过的记录数
        limit: 返回的最大记录数
        
    Returns:
        任务列表
    """
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        tasks = crud.get_all(skip=skip, limit=limit)
        return [task.to_dict() for task in tasks]
    finally:
        db.close()


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: int):
    """
    删除任务
    
    Args:
        task_id: 任务 ID
        
    Returns:
        删除结果
    """
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        success = crud.delete(task_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"任务 ID {task_id} 不存在")
        return {"message": f"任务 ID {task_id} 已删除", "success": True}
    finally:
        db.close()


@app.post("/api/tasks/search/enhanced-task")
async def search_tasks_by_enhanced_task(request: TaskSearch):
    """
    根据增强任务文本进行向量相似度搜索
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果列表
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        results = crud.search_by_enhanced_task(
            query_text=request.query_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return results
    finally:
        db.close()


@app.get("/api/tasks/count")
async def count_tasks():
    """
    获取任务存储中的总记录数
    
    Returns:
        总记录数
    """
    db = next(get_task_db())
    try:
        crud = TaskStorageCRUD(db)
        count = crud.count()
        return {"count": count}
    finally:
        db.close()


# ==================== 业务知识库 API ====================

@app.post("/api/business-knowledge")
async def create_business_knowledge(request: BusinessKnowledgeCreate):
    """
    创建业务知识条目
    
    Args:
        request: 业务知识创建请求
        
    Returns:
        创建的知识条目
    """
    if not request.question_text or not request.question_text.strip():
        raise HTTPException(status_code=400, detail="问题文本不能为空")
    if not request.answer_text or not request.answer_text.strip():
        raise HTTPException(status_code=400, detail="答案文本不能为空")
    
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        knowledge = crud.create(
            question_text=request.question_text,
            answer_text=request.answer_text
        )
        return knowledge.to_dict()
    finally:
        db.close()


@app.get("/api/business-knowledge/{knowledge_id}")
async def get_business_knowledge(knowledge_id: int):
    """
    获取业务知识条目详情
    
    Args:
        knowledge_id: 知识条目 ID
        
    Returns:
        知识条目详情
    """
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        knowledge = crud.get_by_id(knowledge_id)
        if not knowledge:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return knowledge.to_dict()
    finally:
        db.close()


@app.get("/api/business-knowledge")
async def list_business_knowledge(skip: int = 0, limit: int = 100):
    """
    获取业务知识列表（分页）
    
    Args:
        skip: 跳过的记录数
        limit: 返回的最大记录数
        
    Returns:
        知识条目列表
    """
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        knowledge_list = crud.get_all(skip=skip, limit=limit)
        return [knowledge.to_dict() for knowledge in knowledge_list]
    finally:
        db.close()


@app.put("/api/business-knowledge/{knowledge_id}")
async def update_business_knowledge(knowledge_id: int, request: BusinessKnowledgeUpdate):
    """
    更新业务知识条目
    
    Args:
        knowledge_id: 知识条目 ID
        request: 更新请求
        
    Returns:
        更新后的知识条目
    """
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        updated_knowledge = crud.update(
            knowledge_id=knowledge_id,
            question_text=request.question_text,
            answer_text=request.answer_text
        )
        if not updated_knowledge:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return updated_knowledge.to_dict()
    finally:
        db.close()


@app.delete("/api/business-knowledge/{knowledge_id}")
async def delete_business_knowledge(knowledge_id: int):
    """
    删除业务知识条目
    
    Args:
        knowledge_id: 知识条目 ID
        
    Returns:
        删除结果
    """
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        success = crud.delete(knowledge_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return {"message": f"知识条目 ID {knowledge_id} 已删除", "success": True}
    finally:
        db.close()


@app.post("/api/business-knowledge/search/question")
async def search_business_knowledge_by_question(request: BusinessKnowledgeSearch):
    """
    根据问题文本搜索业务知识
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果列表
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        results = crud.search_by_question(
            query_text=request.query_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return results
    finally:
        db.close()


@app.post("/api/business-knowledge/search/answer")
async def search_business_knowledge_by_answer(request: BusinessKnowledgeSearch):
    """
    根据答案文本搜索业务知识
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果列表
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        results = crud.search_by_answer(
            query_text=request.query_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return results
    finally:
        db.close()


@app.get("/api/business-knowledge/count")
async def count_business_knowledge():
    """
    获取业务知识库中的总记录数
    
    Returns:
        总记录数
    """
    db = next(get_business_db())
    try:
        crud = BusinessKnowledgeCRUD(db)
        count = crud.count()
        return {"count": count}
    finally:
        db.close()


# ==================== 推理知识库 API ====================

@app.post("/api/reasoning-knowledge")
async def create_reasoning_knowledge(request: ReasoningKnowledgeCreate):
    """
    创建推理知识条目
    
    Args:
        request: 推理知识创建请求
        
    Returns:
        创建的知识条目
    """
    if not request.task_text or not request.task_text.strip():
        raise HTTPException(status_code=400, detail="任务文本不能为空")
    if not request.step_text or not request.step_text.strip():
        raise HTTPException(status_code=400, detail="步骤文本不能为空")
    
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        knowledge = crud.create(
            task_text=request.task_text,
            step_text=request.step_text
        )
        return knowledge.to_dict()
    finally:
        db.close()


@app.get("/api/reasoning-knowledge/{knowledge_id}")
async def get_reasoning_knowledge(knowledge_id: int):
    """
    获取推理知识条目详情
    
    Args:
        knowledge_id: 知识条目 ID
        
    Returns:
        知识条目详情
    """
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        knowledge = crud.get_by_id(knowledge_id)
        if not knowledge:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return knowledge.to_dict()
    finally:
        db.close()


@app.get("/api/reasoning-knowledge")
async def list_reasoning_knowledge(skip: int = 0, limit: int = 100):
    """
    获取推理知识列表（分页）
    
    Args:
        skip: 跳过的记录数
        limit: 返回的最大记录数
        
    Returns:
        知识条目列表
    """
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        knowledge_list = crud.get_all(skip=skip, limit=limit)
        return [knowledge.to_dict() for knowledge in knowledge_list]
    finally:
        db.close()


@app.put("/api/reasoning-knowledge/{knowledge_id}")
async def update_reasoning_knowledge(knowledge_id: int, request: ReasoningKnowledgeUpdate):
    """
    更新推理知识条目
    
    Args:
        knowledge_id: 知识条目 ID
        request: 更新请求
        
    Returns:
        更新后的知识条目
    """
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        updated_knowledge = crud.update(
            knowledge_id=knowledge_id,
            task_text=request.task_text,
            step_text=request.step_text
        )
        if not updated_knowledge:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return updated_knowledge.to_dict()
    finally:
        db.close()


@app.delete("/api/reasoning-knowledge/{knowledge_id}")
async def delete_reasoning_knowledge(knowledge_id: int):
    """
    删除推理知识条目
    
    Args:
        knowledge_id: 知识条目 ID
        
    Returns:
        删除结果
    """
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        success = crud.delete(knowledge_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"知识条目 ID {knowledge_id} 不存在")
        return {"message": f"知识条目 ID {knowledge_id} 已删除", "success": True}
    finally:
        db.close()


@app.post("/api/reasoning-knowledge/search/task")
async def search_reasoning_knowledge_by_task(request: ReasoningKnowledgeSearch):
    """
    根据任务文本搜索推理知识
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果列表
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        results = crud.search_by_task(
            query_text=request.query_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return results
    finally:
        db.close()


@app.post("/api/reasoning-knowledge/search/step")
async def search_reasoning_knowledge_by_step(request: ReasoningKnowledgeSearch):
    """
    根据步骤文本搜索推理知识
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果列表
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        results = crud.search_by_step(
            query_text=request.query_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return results
    finally:
        db.close()


@app.get("/api/reasoning-knowledge/count")
async def count_reasoning_knowledge():
    """
    获取推理知识库中的总记录数
    
    Returns:
        总记录数
    """
    db = next(get_reasoning_db())
    try:
        crud = ReasoningKnowledgeCRUD(db)
        count = crud.count()
        return {"count": count}
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)