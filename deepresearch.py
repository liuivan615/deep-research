#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import threading
import time  # 添加time模块导入
import queue  # 添加队列模块用于线程间通信
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import urllib.request
import urllib.parse
import sys
import os
import uuid
import re  # 添加正则表达式模块
import numpy as np  # 导入numpy用于向量计算
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
    # 延迟全局懒加载SentenceTransformer模型
    ST_MODEL = None
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    ST_MODEL = None
    print("警告: SentenceTransformer库未安装，将使用备选嵌入方法。可通过 'pip install sentence-transformers' 安装。", file=sys.stderr)
from search_mcp import init_llm_client, generate_query, web_search, process_link, get_new_search_queries, get_images, generate_narrative
import prompts

# 全局状态：日志和最终报告内容
logs = []
final_report = ""
is_running = False
clarification_questions = []  # 存储澄清问题
clarification_answers = {}    # 存储用户对澄清问题的回答
initial_analysis = ""         # 存储初步分析
current_stage = "initial"     # 当前研究阶段

# 添加日志锁，防止多线程日志乱序
log_lock = threading.Lock()

# 历史记录存储
history_file = "research_history.json"
research_history = []

# 加载历史记录
def load_history():
    global research_history
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                research_history = json.load(f)
    except Exception as e:
        print(f"加载历史记录失败: {e}", file=sys.stderr)
        research_history = []

# 保存历史记录
def save_history():
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(research_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存历史记录失败: {e}", file=sys.stderr)

# 添加历史记录
def add_history_item(research_id, topic, model, timestamp=None):
    if timestamp is None:
        timestamp = int(time.time())
    
    history_item = {
        "id": research_id,
        "topic": topic,
        "model": model,
        "timestamp": timestamp
    }
    
    research_history.append(history_item)
    save_history()
    return history_item

def log(message: str):
    """添加一条日志消息（第一人称叙述）到日志列表"""
    global logs
    with log_lock:
        logs.append(message)
        # 控制台调试用，flush=True确保立即输出
        print(message, file=sys.stderr, flush=True)

# 模型配置：远程API模型和本地Ollama模型（请将API Key替换为您自己的）
MODELS = {
    # 在线模型配置
    "grok-3": {
        "base_url": "https://api.qingtian.shop/v1",
        "api_key": "sk-kTAzpbLula7HSbwhrdKRHnB2ZssCYlftXoXEvSHy",  # 替换为您的API密钥
        "real_name": "grok-3"
    },
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为您的API密钥
        "real_name": "deepseek-chat"
    },
    "deepseek-reasoner": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为您的API密钥
        "real_name": "deepseek-reasoner"
    },
    "gpt-4o": {
        "base_url": "https://api.nuwaapi.com/v1",
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为您的API密钥
        "real_name": "gpt-4o"
    },
    # 本地 Ollama 模型配置
    "ollama_deepseek_r1_8b": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # 本地模型默认值
        "real_name": "deepseek-r1:8b",
        "is_ollama": True
    },
    "ollama_deepseek_r1_14b": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "deepseek-r1:14b",
        "is_ollama": True
    },
    "ollama_mistral": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "mistral:latest",
        "is_ollama": True
    },
    "ollama_qwen7b_distill": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "cyberuser42/DeepSeek-R1-Distill-Qwen-7b:latest",
        "is_ollama": True
    },
    "ollama_qwen14b_distill": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "cyberuser42/DeepSeek-R1-Distill-Qwen-14B:latest",
        "is_ollama": True
    },
    "ollama_gemma_12b": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "gemma:12b",
        "is_ollama": True
    },
    "ollama_gemma3": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "gemma3:latest",
        "is_ollama": True
    },
    "ollama_llama3": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "llama3.2:1b-instruct-fp16",
        "is_ollama": True
    },
    "ollama_qwq": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "qwq:latest",
        "is_ollama": True
    },
    # 添加检索专用小模型
    "ollama_qwen3_06b": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "real_name": "qwen3:0.6b",
        "is_ollama": True
    }
}

class LLMClient:
    """通用LLM客户端，支持远程API和本地Ollama模型"""
    def __init__(self, model_key: str):
        if model_key not in MODELS:
            raise ValueError(f"未知的模型密钥: {model_key}")
        cfg = MODELS[model_key]
        self.model_key = model_key
        self.model = cfg.get("real_name", model_key)
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg["api_key"]
        self.is_ollama = cfg.get("is_ollama", False)
        # 准备HTTP请求头
        self.headers = {"Content-Type": "application/json"}
        if not self.is_ollama:
            # 远程API使用Bearer认证
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        # 本地Ollama通常不需要认证

    def ask(self, messages, temperature=0.7, max_tokens=1500, max_retries=2):
        """发送聊天消息给LLM并返回助手回复内容"""
        for retry in range(max_retries + 1):
            try:
                if not self.is_ollama:
                    # 调用远程OpenAI兼容API
                    url = f"{self.base_url}/chat/completions"
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    data = self._post_json(url, payload)
                    if data and "choices" in data and data["choices"]:
                        content = data["choices"][0]["message"].get("content", "")
                        return content.strip()
                    else:
                        if retry < max_retries:
                            print(f"意外的API响应，重试 ({retry+1}/{max_retries+1})...", file=sys.stderr)
                            time.sleep(5 * (retry + 1))  # 递增等待时间，从5秒开始
                            continue
                        raise Exception(f"Unexpected response: {data}")
                else:
                    # 调用本地Ollama API
                    prompt_text = self._format_ollama_prompt(messages)
                    api_base = self.base_url.rsplit("/v", 1)[0]  # 去掉 /v1
                    url = f"{api_base}/api/generate"
                    payload = {
                        "model": self.model,
                        "prompt": prompt_text,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    }
                    data = self._post_json(url, payload)
                    if data and "response" in data:
                        content = data["response"]
                        return content.strip()
                    else:
                        if retry < max_retries:
                            print(f"意外的Ollama响应，重试 ({retry+1}/{max_retries+1})...", file=sys.stderr)
                            time.sleep(5 * (retry + 1))  # 更长的等待时间
                            continue
                        raise Exception(f"Unexpected Ollama response: {data}")
            except Exception as e:
                if retry < max_retries:
                    print(f"LLM调用出错: {e}，重试 ({retry+1}/{max_retries+1})...", file=sys.stderr)
                    time.sleep(10 * (retry + 1))  # 更长的等待时间，从10秒开始
                    continue
                # 将最后一次异常抛给调用者处理
                raise

    def _format_ollama_prompt(self, messages):
        """将消息列表格式化为Ollama所需的prompt字符串"""
        system_content = ""
        conversation = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                conversation += f"[用户]\n{content}\n\n"
            elif role == "assistant":
                conversation += f"[助手]\n{content}\n\n"
        if system_content:
            conversation = f"[系统指令]\n{system_content}\n\n" + conversation
        # 对话结束，等待助手回答
        conversation += "[助手]\n"
        return conversation

    def _post_json(self, url, payload):
        """发送JSON POST请求并返回解析后的JSON响应"""
        data_bytes = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data_bytes, headers=self.headers, method="POST")
        # 增加超时时间到300秒
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp_data = resp.read()
            try:
                return json.loads(resp_data.decode('utf-8', errors='ignore'))
            except json.JSONDecodeError:
                return None

# 创建研究结果目录
def ensure_research_dir():
    research_dir = "research_data"
    if not os.path.exists(research_dir):
        os.makedirs(research_dir)
    return research_dir

# 保存研究报告和日志
def save_research_data(research_id, topic, model, logs_data, report_data, clarification_data=None, analysis_data=None):
    try:
        research_dir = ensure_research_dir()
        research_path = os.path.join(research_dir, f"{research_id}.json")
        
        research_data = {
            "id": research_id,
            "topic": topic,
            "model": model,
            "timestamp": int(time.time()),
            "logs": logs_data,
            "report": report_data,
            "clarification_questions": clarification_data or [],
            "clarification_answers": clarification_answers or {},
            "initial_analysis": analysis_data or ""
        }
        
        with open(research_path, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"保存研究数据失败: {e}", file=sys.stderr)
        return False

# 加载研究数据
def load_research_data(research_id):
    try:
        research_dir = ensure_research_dir()
        research_path = os.path.join(research_dir, f"{research_id}.json")
        
        if not os.path.exists(research_path):
            return None
            
        with open(research_path, 'r', encoding='utf-8') as f:
            research_data = json.load(f)
            
        return research_data
    except Exception as e:
        print(f"加载研究数据失败: {e}", file=sys.stderr)
        return None

def generate_clarification_questions(topic, llm_client):
    """生成3-5个针对研究主题的澄清问题"""
    try:
        # 使用CLARIFICATION_PROMPT模板
        prompt = prompts.CLARIFICATION_PROMPT.format(topic)
        
        # 调用LLM生成问题
        response = llm_client.ask([
            {"role": "system", "content": "你是一个专业的研究助手，擅长提出有助于理解用户需求的澄清性问题。"},
            {"role": "user", "content": prompt}
        ], temperature=0.7)
        
        # 解析响应，提取问题
        if response:
            # 尝试从回复中提取编号问题
            import re
            questions = []
            
            # 匹配类似 "1. 问题内容" 或 "问题1：问题内容" 的模式
            patterns = [
                r'\d+\.\s*(.*?)(?=\d+\.|$)',  # 匹配 "1. 问题" 格式
                r'问题\s*\d+\s*[:：]\s*(.*?)(?=问题\s*\d+|$)',  # 匹配 "问题1：问题内容" 格式
                r'[\n\r]+([^1234567890\n\r].*?)(?=[\n\r]+|$)'  # 匹配段落，作为后备
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    # 清理匹配结果
                    cleaned_matches = [m.strip() for m in matches if m.strip() and len(m.strip()) > 10]
                    if cleaned_matches:
                        questions = cleaned_matches
                        break
            
            # 如果正则表达式提取失败，简单按行分割
            if not questions:
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    # 只保留看起来像问题的行
                    if line and len(line) > 10 and not line.startswith('我想') and '?' in line or '？' in line:
                        questions.append(line)
            
            # 清理并限制问题数量在3-5个
            final_questions = []
            for q in questions:
                # 删除可能的数字前缀
                q = re.sub(r'^\d+\.\s*', '', q)
                # 简单去除非问题的行
                if len(q) > 15 and not q.startswith('请回答') and not q.startswith('您的回答'):
                    final_questions.append(q)
            
            # 确保问题数量在3-5个
            if len(final_questions) > 5:
                final_questions = final_questions[:5]
            elif len(final_questions) < 3:
                # 如果提取的问题不足3个，使用完整响应
                return [response.strip()]
            
            return final_questions
        else:
            # 如果LLM没有响应，返回一个通用问题
            return [f"您能详细说明一下您对'{topic}'研究的具体需求吗？"]
    except Exception as e:
        print(f"生成澄清问题时出错: {e}", file=sys.stderr)
        # 出错时返回一个基本问题
        return [f"您希望了解'{topic}'的哪些具体方面？"]

def generate_initial_analysis(topic, clarification_answers, llm_client):
    """根据主题和用户的澄清问题回答，生成500字的初步分析"""
    try:
        # 准备用户回答的文本
        answers_text = ""
        for i, answer in clarification_answers.items():
            answers_text += f"问题 {int(i)+1} 的回答: {answer}\n\n"
        
        # 使用INITIAL_ANALYSIS_PROMPT模板
        prompt = prompts.INITIAL_ANALYSIS_PROMPT.format(topic, answers_text)
        
        # 调用LLM生成分析
        response = llm_client.ask([
            {"role": "system", "content": "你是一个专业的研究分析师，善于基于有限信息进行初步分析。"},
            {"role": "user", "content": prompt}
        ], temperature=0.7, max_tokens=2000)
        
        if response:
            return response.strip()
        else:
            # 如果LLM没有响应，返回一个通用分析
            return f"关于'{topic}'的研究将围绕用户提供的关键信息展开。我会搜集相关资料，全面分析该主题的各个方面，为用户提供有价值的见解和建议。"
    except Exception as e:
        print(f"生成初步分析时出错: {e}", file=sys.stderr)
        # 出错时返回一个基本分析
        return f"我将开始对'{topic}'进行研究。基于现有信息，我会系统地搜集和分析相关数据，提供全面的研究报告。"

def start_clarification_phase(topic, model_key, research_id):
    """开始澄清阶段，生成澄清问题"""
    global current_stage, clarification_questions
    
    log(f"在开始深入研究之前，我想先更好地了解您对'{topic}'的具体需求。")
    
    # 初始化LLM客户端
    llm_client = LLMClient(model_key)
    init_llm_client(llm_client)
    
    # 生成澄清问题
    clarification_questions = generate_clarification_questions(topic, llm_client)
    
    # 记录问题
    log("为了更好地理解您的需求，请回答以下问题：")
    for i, question in enumerate(clarification_questions):
        log(f"问题 {i+1}: {question}")
    
    log("请在界面中回答这些问题，您的回答将帮助我提供更精确的研究内容。")
    
    # 更新阶段
    current_stage = "clarification"
    
    # 保存数据
    save_research_data(research_id, topic, model_key, logs, "", clarification_questions)
    
    return research_id

def submit_clarification_answers(research_id, answers):
    """提交用户对澄清问题的回答，并开始初步分析"""
    global clarification_answers, current_stage
    
    # 加载研究数据
    research_data = load_research_data(research_id)
    if not research_data:
        return {"error": "未找到研究数据"}
    
    topic = research_data.get("topic", "")
    model = research_data.get("model", "")
    clarification_questions = research_data.get("clarification_questions", [])
    
    # 保存用户回答
    clarification_answers = answers
    
    # 初始化LLM客户端
    llm_client = LLMClient(model)
    init_llm_client(llm_client)
    
    # 记录回答
    log("感谢您的回答，这将帮助我更好地定制研究内容。")
    for i, question in enumerate(clarification_questions):
        answer = answers.get(str(i), "未提供回答")
        log(f"问题 {i+1}: {question}")
        log(f"您的回答: {answer}")
    
    # 更新阶段
    current_stage = "analysis"
    
    # 继续到初步分析阶段
    return start_analysis_phase(topic, model, research_id)

def start_analysis_phase(topic, model_key, research_id):
    """开始初步分析阶段，生成500字的分析"""
    global current_stage, initial_analysis
    
    log("基于您的研究主题和回答，我现在将进行初步分析...")
    
    # 初始化LLM客户端
    llm_client = LLMClient(model_key)
    init_llm_client(llm_client)
    
    # 生成初步分析
    initial_analysis = generate_initial_analysis(topic, clarification_answers, llm_client)
    
    # 记录分析
    log("以下是我对研究主题的初步分析：")
    log(initial_analysis)
    
    # 更新阶段
    current_stage = "research"
    
    # 保存数据
    save_research_data(research_id, topic, model_key, logs, "", clarification_questions, initial_analysis)
    
    # 开始正式研究
    return continue_research(topic, model_key, research_id)

def continue_research(topic, model_key, research_id):
    """继续进行正式的研究流程"""
    # 启动后台线程执行研究
    research_thread = threading.Thread(
        target=run_research, 
        args=(topic, model_key, research_id)
    )
    research_thread.daemon = True
    research_thread.start()
    
    return research_id

# 全局变量
retriever = None  # 智能检索系统实例，会在run_research时初始化
filter_queue = queue.Queue()  # 待过滤文本队列
filtered_results = {}  # 存储过滤后的结果，键为文本ID
filter_lock = threading.Lock()  # 过滤结果访问锁

def run_research(topic: str, model_key: str, research_id=None):
    """在后台线程中执行深度研究流程"""
    global final_report, is_running, logs, current_stage, retriever
    
    # 如果当前不是研究阶段，说明还没完成前面的澄清和分析阶段
    if current_stage != "research":
        return research_id
    
    # 如果没有提供研究ID，生成一个新的
    if research_id is None:
        research_id = str(uuid.uuid4())
    
    is_running = True
    current_topic = topic
    current_model = model_key
    
    # 创建并启动并行文本过滤线程
    filter_thread = None
    try:
        filter_thread = TextFilterThread(topic)
        filter_thread.start()
        log("已启动并行文本过滤线程，使用qwen3:0.6b模型处理信息")
    except Exception as e:
        log(f"启动并行文本过滤线程失败: {e}，将使用主线程处理")
    
    # 初始化过滤任务队列
    run_research.pending_filters = []
    
    try:
        log(f"现在我将基于初步分析，开始深入研究'{topic}'主题...")
        # 初始化LLM客户端
        llm_client = LLMClient(model_key)
        init_llm_client(llm_client)  # 将LLM客户端提供给搜索模块
        
        # 主流程使用的检索系统仍然初始化，用于主LLM的语义理解
        retriever = SmartRetriever()
        log("已初始化主流程智能检索系统")
        
        # 第1步：生成初始搜索查询
        narrative = generate_narrative("准备开始研究", f"需要对主题'{topic}'进行初步搜索", topic)
        log(narrative)
        
        try:
            query_list_str = generate_query(topic)
            try:
                # 首先尝试eval
                initial_queries = eval(query_list_str) if query_list_str else []
            except Exception:
                # eval失败时尝试使用正则表达式提取
                import re
                pattern = r"[\'\"]([^\'\"]*)[\'\"]"
                matches = re.findall(pattern, query_list_str)
                initial_queries = matches if matches else []
                if not matches:
                    # 如果还是无法提取，则直接按逗号拆分
                    initial_queries = [q.strip() for q in query_list_str.split(',') if q.strip()]
        except Exception as e:
            log(f"生成初始搜索查询时遇到问题: {e}")
            initial_queries = []
        if not initial_queries or not isinstance(initial_queries, list):
            initial_queries = [topic]
        if topic not in initial_queries:
            initial_queries.append(topic)
        
        search_terms = ", ".join(initial_queries)
        narrative = generate_narrative("计划搜索关键词", f"已确定以下搜索关键词: {search_terms}", topic)
        log(narrative)
        
        all_search_queries = initial_queries.copy()
        aggregated_contexts = []
        
        # 初始化new_search_queries变量
        new_search_queries = []
        
        # 最多进行3轮搜索迭代
        for iteration in range(1, 4):
            log(f"\n---- 第{iteration}轮搜索 ----")
            
            iteration_narrative = generate_narrative(
                f"开始第{iteration}轮搜索", 
                f"这是第{iteration}轮搜索，{'将进行初步广泛搜索' if iteration == 1 else '将深入探索特定方面' if iteration == 2 else '将填补信息空缺'}", 
                topic
            )
            log(iteration_narrative)
            
            iteration_contexts = []
            unique_links = {}
            
            # 对每个查询执行搜索
            current_queries = initial_queries if iteration == 1 else new_search_queries
            
            for q_index, q in enumerate(current_queries):
                search_narrative = generate_narrative("执行搜索", f"正在搜索关键词: {q}", topic)
                log(search_narrative)
                
                # 避免过于频繁的搜索请求
                if q_index > 0:
                    time.sleep(2)  # 每个查询之间添加延时
                
                # 增加重试和等待时间，避免超时
                max_retries = 2
                for retry in range(max_retries):
                    try:
                        results = web_search(q, top_k=3, categories='general')
                        break  # 成功获取结果，跳出重试循环
                    except Exception as e:
                        if retry < max_retries - 1:
                            log(f"搜索'{q}'时发生错误: {e}，正在重试...")
                            time.sleep(3)  # 重试前等待3秒
                        else:
                            log(f"搜索'{q}'失败: {e}")
                            results = []
                
                if results:
                    found_narrative = generate_narrative("找到搜索结果", f"关键词'{q}'返回了{len(results)}个结果", topic)
                    log(found_narrative)
                
                for link in results:
                    if link not in unique_links:
                        unique_links[link] = q
            
            if not unique_links:
                no_results_narrative = generate_narrative("搜索结果为空", "没有找到任何新的相关网页", topic)
                log(no_results_narrative)
                continue  # 继续下一轮搜索，而不是break
            
            # 处理每个唯一链接
            for link_index, (link, search_query) in enumerate(unique_links.items()):
                domain = urllib.parse.urlparse(link).netloc
                reading_narrative = generate_narrative("阅读网页", f"正在查看来自{domain}的网页", topic)
                log(reading_narrative)
                
                # 避免过于频繁的页面请求
                if link_index > 0:
                    time.sleep(3)  # 每个页面处理之间添加延时
                
                try:
                    context = process_link(link, topic, search_query)
                    if context:
                        # 提取源信息
                        source_info = ""
                        if "来源:" in context:
                            source_info = "来源:" + context.split("来源:")[1]
                        
                        # 使用并行过滤机制处理文本
                        log(f"将从{domain}获取的内容提交给并行过滤线程...")
                        
                        # 提交到并行过滤队列
                        text_id = filter_text_parallel(context, topic, source_info)
                        
                        # 在主线程中继续处理其他内容，之后再获取结果
                        pending_filters = getattr(run_research, "pending_filters", [])
                        pending_filters.append((text_id, link))
                        run_research.pending_filters = pending_filters
                        
                        # 更新用户界面
                        log(f"已将{domain}的内容提交给qwen3:0.6b模型进行并行过滤，处理ID:{text_id[:8]}")
                    else:
                        # 确保在日志中显示域名和链接
                        no_info_narrative = generate_narrative("页面无关", f"检查了{domain}({link})但未找到相关信息", topic)
                        log(no_info_narrative)
                except Exception as e:
                    log(f"处理链接 {link} 时出错: {e}")
            
            # 收集并行过滤的结果
            log("正在收集并行文本过滤的结果...")
            pending_filters = getattr(run_research, "pending_filters", [])
            
            if pending_filters:
                # 等待所有处理完成，但最多等待30秒
                wait_start = time.time()
                collected_count = 0
                
                while pending_filters and time.time() - wait_start < 30:
                    for i in range(len(pending_filters)-1, -1, -1):  # 倒序遍历，方便删除
                        text_id, link = pending_filters[i]
                        filtered_text = get_filtered_text(text_id, timeout=0)  # 不等待，立即检查
                        
                        if filtered_text:
                            # 收集过滤后的内容
                            domain = urllib.parse.urlparse(link).netloc
                            iteration_contexts.append(filtered_text)
                            
                            # 生成摘要信息
                            context_text = filtered_text.split('来源:')[0].strip() if '来源:' in filtered_text else filtered_text
                            summary = context_text.split('\n')[0]
                            if len(summary) > 100:
                                summary = summary[:100] + '...'
                            
                            # 记录信息
                            found_info_narrative = generate_narrative("已完成过滤", f"从{domain}过滤出高质量信息", topic)
                            log(found_info_narrative)
                            log(f"摘要: {summary}")
                            
                            # 从待处理列表中移除
                            del pending_filters[i]
                            collected_count += 1
                    
                    # 短暂等待后继续检查
                    if pending_filters:
                        time.sleep(1)
                
                # 更新待处理列表
                run_research.pending_filters = pending_filters
                
                # 如果还有未处理完的，记录日志
                if pending_filters:
                    log(f"有{len(pending_filters)}个文本过滤任务尚未完成，将在下一轮检查")
                
                log(f"本轮已收集{collected_count}个过滤后的文本")
            
            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
                summary_narrative = generate_narrative(
                    "总结搜索成果", 
                    f"第{iteration}轮搜索找到了{len(iteration_contexts)}条有价值的信息", 
                    topic
                )
                log(summary_narrative)
            else:
                no_value_narrative = generate_narrative("无收获", f"第{iteration}轮搜索未找到有价值的信息", topic)
                log(no_value_narrative)
            
            # 自我反思，判断是否需要继续（最后一轮无需反思）
            if iteration < 3:
                reflection_narrative = generate_narrative("反思当前信息", "评估已获取的信息，确定下一步搜索方向", topic)
                log(reflection_narrative)
                
                try:
                    max_retries = 2
                    for retry in range(max_retries):
                        try:
                            new_search_queries = get_new_search_queries(topic, all_search_queries, aggregated_contexts)
                            break  # 成功获取结果
                        except Exception as e:
                            if retry < max_retries - 1:
                                log(f"分析搜索方向时出错，正在重试...")
                                time.sleep(2)
                            else:
                                log(f"分析是否需要更多搜索时出错: {e}")
                                new_search_queries = []
                except Exception as e:
                    log(f"分析是否需要更多搜索时出错: {e}")
                    new_search_queries = []
                
                if new_search_queries:
                    new_search_queries = [q for q in new_search_queries if q not in all_search_queries]
                
                if new_search_queries:
                    all_search_queries.extend(new_search_queries)
                    new_terms = ", ".join(new_search_queries)
                    next_search_narrative = generate_narrative("计划下一轮搜索", f"确定了新的搜索关键词: {new_terms}", topic)
                    log(next_search_narrative)
                else:
                    enough_info_narrative = generate_narrative("决定结束搜索", "已收集足够的信息，无需继续搜索", topic)
                    log(enough_info_narrative)
                    break
            
            # 轮次之间添加延时，避免过于频繁的请求
            time.sleep(5)
        
        # 搜索完成后，获取相关图片
        images_narrative = generate_narrative("搜索相关图片", "在文字资料之外寻找视觉内容", topic)
        log(images_narrative)
        
        try:
            max_retries = 2
            for retry in range(max_retries):
                try:
                    images_result = get_images(topic)
                    break  # 成功获取结果
                except Exception as e:
                    if retry < max_retries - 1:
                        log(f"获取图片时出错，正在重试...")
                        time.sleep(3)
                    else:
                        log(f"获取图片失败: {e}")
                        images_result = {}
        except Exception as e:
            log(f"获取图片时出错: {e}")
            images_result = {}
        
        images_dict = {}
        if isinstance(images_result, dict):
            images_dict = images_result
            if images_dict:
                images_found_narrative = generate_narrative("找到图片", f"找到了{len(images_dict)}张相关图片", topic)
                log(images_found_narrative)
            else:
                no_images_narrative = generate_narrative("无图片", "未找到相关图片", topic)
                log(no_images_narrative)
        else:
            if images_result:
                log(str(images_result))
        
        # 整理最终报告
        report_narrative = generate_narrative("准备生成报告", f"已收集全面信息，开始撰写关于'{topic}'的研究报告", topic)
        log(report_narrative)
        
        # 收集任何剩余的过滤结果
        pending_filters = getattr(run_research, "pending_filters", [])
        if pending_filters:
            log(f"正在等待剩余的{len(pending_filters)}个文本过滤任务完成...")
            
            # 等待所有处理完成，最多等待60秒
            wait_start = time.time()
            while pending_filters and time.time() - wait_start < 60:
                for i in range(len(pending_filters)-1, -1, -1):  # 倒序遍历
                    text_id, link = pending_filters[i]
                    filtered_text = get_filtered_text(text_id, timeout=0)
                    
                    if filtered_text:
                        # 收集过滤后的内容
                        domain = urllib.parse.urlparse(link).netloc
                        aggregated_contexts.append(filtered_text)
                        
                        # 记录简短信息
                        log(f"收集了来自{domain}的最终过滤结果")
                        
                        # 从待处理列表中移除
                        del pending_filters[i]
                
                # 短暂等待后继续检查
                if pending_filters:
                    time.sleep(1)
            
            # 更新待处理列表
            run_research.pending_filters = pending_filters
            
            # 如果还有未处理完的，使用超时方式强制获取
            if pending_filters:
                log(f"仍有{len(pending_filters)}个任务未完成，将强制获取结果")
                for text_id, link in pending_filters:
                    filtered_text = get_filtered_text(text_id, timeout=3)  # 短超时
                    if filtered_text:
                        aggregated_contexts.append(filtered_text)
        
        # 停止过滤线程
        if filter_thread and filter_thread.is_alive():
            log("正在停止文本过滤线程...")
            filter_thread.stop()
            filter_thread.join(3)  # 等待线程退出，最多3秒
            log("文本过滤线程已停止")
        
        collected_info = '\n\n---\n\n'.join(aggregated_contexts)
        if images_dict:
            collected_info += "\n\n相关图片：\n"
            for url, desc in images_dict.items():
                desc_text = desc or "(无描述)"
                collected_info += f"- {desc_text} ({url})\n"
        
        # 准备澄清回答的内容
        clarification_content = ""
        for i, question in enumerate(clarification_questions):
            answer = clarification_answers.get(str(i), "用户未提供回答")
            clarification_content += f"问题 {i+1}: {question}\n回答: {answer}\n\n"
        
        # 使用智能检索系统处理收集到的信息
        log("正在进行最终信息整合和优化，确保内容与研究主题高度相关...")
        try:
            # 确保retriever已经初始化
            if retriever is None:
                retriever = SmartRetriever()
                
            # 对所有收集到的信息进行一次整体优化
            filtered_info = retriever.smart_retrieve(collected_info, topic)
            
            if filtered_info and len(filtered_info) > len(collected_info) * 0.3:  # 确保过滤不会过度删减内容
                reduction = 100 - (len(filtered_info) * 100 // max(1, len(collected_info)))
                retrieval_narrative = generate_narrative("最终优化完成", f"将{len(collected_info)}字符的原始信息优化为{len(filtered_info)}字符的高质量内容，减少{reduction}%", topic)
                log(retrieval_narrative)
                
                # 使用优化后的信息
                optimized_info = filtered_info
            else:
                log("最终优化返回的内容比例过低，将使用原始收集的信息以保证内容丰富度")
                optimized_info = collected_info
        except Exception as e:
            log(f"最终信息优化处理失败: {e}，将使用原始收集的信息")
            optimized_info = collected_info
        
        # 生成最终报告
        try:
            log("正在分段生成最终研究报告，按照学术论文格式组织内容...")
            final_report_parts = []
            
            # 第一步：确定报告合适的结构（基于主题和收集的信息）
            structure_prompt = f"""
            分析主题"{topic}"和用户需求，确定最合适的研究报告结构。
            
            主题: {topic}
            收集的信息摘要: {collected_info[:1500] if collected_info else "无详细信息"}
            用户澄清: {clarification_content[:500] if clarification_content else "无澄清信息"}
            
            请从以下基础结构出发，选择最适合的章节组成，可以添加、删除或修改章节：
            
            1. 摘要
            2. 引言（研究背景、问题陈述、研究目的）
            3. 文献综述/理论基础
            4. 研究方法/分析框架
            5. 数据分析与发现（可分为多个子章节）
            6. 讨论（结果解释、对比分析）
            7. 结论与建议
            8. 未来展望
            9. 参考资料
            
            对于非学术主题，可考虑以下替代结构：
            
            1. 摘要/概述
            2. 背景介绍
            3. 现状分析
            4. 关键挑战/机遇
            5. 案例研究
            6. 趋势预测
            7. 实用建议
            8. 结论
            
            请输出你选择的章节结构（只需列出章节标题和简短描述），不要解释选择原因。每个章节需要注明预计字数（例如：1000-1500字）。
            """
            
            try:
                structure_response = llm_client.ask([{"role": "user", "content": structure_prompt}], temperature=0.4, max_tokens=1500)
                log("已根据主题特性确定报告结构框架")
            except Exception as e:
                log(f"确定报告结构时出错: {e}，将使用默认结构")
                structure_response = """
                1. 摘要 (800-1200字)
                2. 引言 (1500-2000字)
                3. 背景分析 (2000-2500字)
                4. 主要发现 (3000-4000字)
                5. 讨论 (2000-2500字)
                6. 结论与建议 (2000-2500字)
                """
            
            # 解析结构响应，提取章节信息
            import re
            chapter_pattern = r'(\d+)\.\s*(.*?)(?:\s*\((\d+)[\-–—](\d+)字\))?\s*(?=$|\n\d+\.)'
            chapters = []
            
            try:
                matches = re.findall(chapter_pattern, structure_response, re.DOTALL)
                for match in matches:
                    chapter_num, chapter_title = match[0], match[1].strip()
                    min_words = int(match[2]) if len(match) > 2 and match[2].isdigit() else 1500
                    max_words = int(match[3]) if len(match) > 3 and match[3].isdigit() else 2500
                    chapters.append({
                        "number": chapter_num,
                        "title": chapter_title,
                        "min_words": min_words,
                        "max_words": max_words
                    })
            except Exception as e:
                log(f"解析章节结构时出错: {e}，将使用默认章节")
                # 默认章节结构
                chapters = [
                    {"number": "1", "title": "摘要", "min_words": 800, "max_words": 1200},
                    {"number": "2", "title": "引言", "min_words": 1500, "max_words": 2000},
                    {"number": "3", "title": "背景分析", "min_words": 2000, "max_words": 2500},
                    {"number": "4", "title": "主要发现", "min_words": 3000, "max_words": 4000},
                    {"number": "5", "title": "讨论", "min_words": 2000, "max_words": 2500},
                    {"number": "6", "title": "结论与建议", "min_words": 2000, "max_words": 2500}
                ]
            
            if not chapters:
                log("无法识别章节结构，将使用默认章节")
                # 同上默认章节结构
                chapters = [
                    {"number": "1", "title": "摘要", "min_words": 800, "max_words": 1200},
                    {"number": "2", "title": "引言", "min_words": 1500, "max_words": 2000},
                    {"number": "3", "title": "背景分析", "min_words": 2000, "max_words": 2500},
                    {"number": "4", "title": "主要发现", "min_words": 3000, "max_words": 4000},
                    {"number": "5", "title": "讨论", "min_words": 2000, "max_words": 2500},
                    {"number": "6", "title": "结论与建议", "min_words": 2000, "max_words": 2500}
                ]
            
            # 记录章节结构
            log(f"确定的报告结构包含{len(chapters)}个章节:")
            for ch in chapters:
                log(f"{ch['number']}. {ch['title']} ({ch['min_words']}-{ch['max_words']}字)")
            
            # 第二步：为每个章节准备定制化提示
            # 根据章节标题生成合适的提示模板
            def get_chapter_prompt(chapter, topic, info_segment, clarification_content):
                chapter_title = chapter["title"].lower()
                min_words = chapter["min_words"]
                max_words = chapter["max_words"]
                
                base_prompt = f"请为研究主题'{topic}'撰写'{chapter['title']}'部分，长度约{min_words}-{max_words}字。"
                
                # 根据章节类型定制提示
                if "摘要" in chapter_title or "概述" in chapter_title:
                    prompt = base_prompt + f"""
                    摘要应简明扼要地概括整个研究的核心内容，包括研究目的、方法、主要发现和结论。
                    不需要引用或详细解释，但必须准确反映报告的主要内容和价值。
                    请确保内容全面但精炼，字数控制在{min_words}-{max_words}之间。
                    """
                
                elif "引言" in chapter_title or "介绍" in chapter_title or "背景" in chapter_title:
                    prompt = base_prompt + f"""
                    引言部分应介绍研究的背景、重要性和目的。
                    应阐明研究问题或主题的来源、当前相关情况以及研究此主题的价值和意义。
                    可包括简要的历史背景、最新发展趋势、研究空白或争议点以及本研究的切入点。
                    请基于以下信息，但不限于这些信息来撰写：
                    
                    {info_segment[:3000]}
                    
                    用户澄清信息：
                    {clarification_content}
                    """
                
                elif "文献" in chapter_title or "综述" in chapter_title or "理论" in chapter_title:
                    prompt = base_prompt + f"""
                    文献综述应对主题相关的已有研究、理论和观点进行系统性回顾。
                    应包括：主要理论框架、关键研究及其发现、存在的争议或研究缺口。
                    请组织已有知识，展示对主题的深入理解，而非简单罗列文献。
                    请基于以下信息，但也可添加您了解的其他相关文献：
                    
                    {info_segment[:4000]}
                    """
                
                elif "方法" in chapter_title or "框架" in chapter_title:
                    prompt = base_prompt + f"""
                    方法部分应详细说明研究采用的方法、框架或工具。
                    可包括：数据来源、分析方法、研究限制等。
                    请根据主题特性选择合适的方法描述方式，确保清晰易懂。
                    请基于以下信息：
                    
                    {info_segment[:2500]}
                    """
                
                elif "数据" in chapter_title or "分析" in chapter_title or "发现" in chapter_title:
                    prompt = base_prompt + f"""
                    此部分是报告的核心，应详细呈现分析结果和主要发现。
                    应包含具体数据、事实、案例分析，并通过图表描述(如适用)。
                    可分为多个子主题或要点，确保层次清晰、内容详实。
                    请基于以下信息进行全面分析：
                    
                    {info_segment}
                    """
                
                elif "讨论" in chapter_title or "比较" in chapter_title:
                    prompt = base_prompt + f"""
                    讨论部分应对研究发现进行深入解读和讨论。
                    应包括：对研究发现的解释、与现有研究的比较、意外发现的解释、研究局限性。
                    请提供深度思考和见解，而非简单重复研究发现。
                    请基于前面的研究发现和以下信息：
                    
                    {info_segment[-4000:]}
                    """
                
                elif "结论" in chapter_title or "建议" in chapter_title:
                    prompt = base_prompt + f"""
                    结论部分应总结研究的核心发现、意义和价值。
                    应包括：研究主要结论、实践或理论启示、具体可行的建议。
                    确保建议具体、有针对性，而非泛泛而谈。
                    请基于整个研究内容和以下信息：
                    
                    {info_segment[-3000:]}
                    
                    用户澄清信息：
                    {clarification_content}
                    """
                
                elif "展望" in chapter_title or "未来" in chapter_title:
                    prompt = base_prompt + f"""
                    未来展望部分应讨论主题的发展趋势和未来研究方向。
                    应包括：预测未来发展、提出值得进一步研究的问题、技术或应用展望。
                    请基于当前信息进行合理推测和展望，不必过于保守。
                    """
                
                elif "案例" in chapter_title:
                    prompt = base_prompt + f"""
                    案例研究部分应提供1-3个详细的、与主题高度相关的案例分析。
                    每个案例应包括背景、关键事件、结果和经验教训。
                    案例应具有代表性，能够有效说明主题的关键方面。
                    请基于以下信息，也可添加其他知名案例：
                    
                    {info_segment}
                    """
                
                elif "参考" in chapter_title or "资料" in chapter_title:
                    prompt = base_prompt + f"""
                    列出研究中引用的主要参考资料，格式规范，信息完整。
                    可包括：学术论文、书籍、报告、网站、新闻等。
                    请基于研究过程中提及的信息来源：
                    
                    {info_segment[-2000:]}
                    """
                
                else:
                    # 通用章节模板
                    prompt = base_prompt + f"""
                    请根据章节标题'{chapter['title']}'的内涵，撰写与研究主题'{topic}'相关的内容。
                    内容应具体、详实、有深度，避免空泛和重复。
                    请基于以下信息，但不限于这些信息：
                    
                    {info_segment}
                    
                    用户澄清信息：
                    {clarification_content}
                    """
                
                # 添加主题相关性和字数要求的强调
                prompt += f"""
                
                重要要求：
                1. 内容必须与主题'{topic}'高度相关，不要偏离主题
                2. 字数必须在{min_words}-{max_words}之间
                3. 内容必须具体、有深度，避免泛泛而谈
                4. 如使用数据或引用，请确保准确性
                5. 使用清晰的段落结构，必要时可使用小标题划分
                
                请注意，这是报告的第{chapter['number']}章节，总共{len(chapters)}个章节。
                """
                
                return prompt
            
            # 第三步：分段生成每个章节
            total_chinese_chars = 0
            
            # 平均分配信息段落给不同章节
            info_segments = []
            if optimized_info:
                info_length = len(optimized_info)
                segment_size = info_length // len(chapters)
                for i in range(len(chapters)):
                    start = i * segment_size
                    end = (i + 1) * segment_size if i < len(chapters) - 1 else info_length
                    info_segments.append(optimized_info[start:end])
            else:
                info_segments = [""] * len(chapters)
            
            # 逐章节生成内容
            for i, chapter in enumerate(chapters):
                chapter_num = chapter["number"]
                chapter_title = chapter["title"]
                info_segment = info_segments[i] if i < len(info_segments) else ""
                
                log(f"正在生成第{chapter_num}章：{chapter_title}")
                
                # 获取章节定制提示
                chapter_prompt = get_chapter_prompt(chapter, topic, info_segment, clarification_content)
                
                # 最多尝试3次生成每个章节
                chapter_content = None
                max_chapter_retries = 3
                
                for retry in range(max_chapter_retries):
                    try:
                        chapter_response = llm_client.ask([{"role": "user", "content": chapter_prompt}], temperature=0.5, max_tokens=3000)
                        
                        # 验证章节内容
                        if chapter_response and len(chapter_response) > 100:
                            # 检查中文字数
                            chinese_char_count = sum(1 for char in chapter_response if '\u4e00' <= char <= '\u9fff')
                            
                            # 检查是否符合最低字数要求（除了摘要和参考资料）
                            min_requirement = chapter["min_words"] * 0.7  # 允许70%的容错空间
                            
                            if chinese_char_count < min_requirement and not ("摘要" in chapter_title or "参考" in chapter_title):
                                if retry < max_chapter_retries - 1:
                                    log(f"生成的'{chapter_title}'内容不足({chinese_char_count}/{chapter['min_words']}字)，正在重新生成... (第{retry+1}次尝试)")
                                    time.sleep(5 * (retry + 1))
                                    continue
                            
                            # 检查是否包含主题关键词
                            topic_words = set(topic.lower().split())
                            chapter_lower = chapter_response.lower()
                            keywords_found = sum(1 for word in topic_words if word in chapter_lower[:500] and len(word) > 2)
                            
                            if keywords_found < 1 and len(topic_words) > 0 and not ("参考" in chapter_title):
                                if retry < max_chapter_retries - 1:
                                    log(f"生成的'{chapter_title}'与主题相关性不足，正在重新生成... (第{retry+1}次尝试)")
                                    time.sleep(5 * (retry + 1))
                                    continue
                            
                            chapter_content = chapter_response
                            total_chinese_chars += chinese_char_count
                            break
                        else:
                            if retry < max_chapter_retries - 1:
                                log(f"生成'{chapter_title}'失败，正在重试... (第{retry+1}次尝试)")
                                time.sleep(5 * (retry + 1))
                            else:
                                log(f"多次尝试后无法生成'{chapter_title}'")
                                chapter_content = f"[{chapter_title}生成失败]"
                    except Exception as e:
                        if retry < max_chapter_retries - 1:
                            log(f"生成'{chapter_title}'时出错: {e}，正在重试... (第{retry+1}次尝试)")
                            time.sleep(10 * (retry + 1))
                        else:
                            log(f"多次尝试后生成'{chapter_title}'失败: {e}")
                            chapter_content = f"[{chapter_title}生成失败: {e}]"
                
                if chapter_content:
                    # 添加章节标题和内容
                    section_number = f"{chapter_num}" if chapter_num else f"{i+1}"
                    final_report_parts.append(f"## {section_number}. {chapter_title}\n\n{chapter_content}")
                    log(f"'{chapter_title}'章节生成完成，约{sum(1 for char in chapter_content if '\u4e00' <= char <= '\u9fff')}中文字")
                else:
                    final_report_parts.append(f"## {chapter_num}. {chapter_title}\n\n[内容生成失败]")
            
            # 第四步：组合所有章节为最终报告
            if final_report_parts:
                # 添加标题和副标题
                current_time = time.strftime("%Y年%m月%d日")
                title = f"# {topic}研究报告\n\n"
                subtitle = f"*研究完成日期：{current_time}*\n\n"
                
                # 生成目录
                toc = "## 目录\n\n"
                for i, chapter in enumerate(chapters):
                    ch_num = chapter["number"] if chapter["number"] else f"{i+1}"
                    ch_title = chapter["title"]
                    # 创建目录链接（转换为锚点格式）
                    anchor = f"{ch_num}-{ch_title.lower().replace(' ', '-').replace('（', '').replace('）', '').replace('(', '').replace(')', '')}"
                    toc += f"{ch_num}. [{ch_title}](#{anchor})\n"
                toc += "\n---\n\n"
                
                # 组合完整报告
                final_report = title + subtitle + toc + "\n\n".join(final_report_parts)
                
                # 计算报告中文字数
                chinese_char_count = sum(1 for char in final_report if '\u4e00' <= char <= '\u9fff')
                log(f"研究报告生成完毕，共约{chinese_char_count}中文字")
                
                # 验证报告总长度是否达到12000中文字
                if chinese_char_count < 12000:
                    log("报告内容不足12000字，正在生成补充内容...")
                    
                    # 确定哪些章节需要扩展
                    expandable_chapters = []
                    for ch in chapters:
                        if "发现" in ch["title"].lower() or "分析" in ch["title"].lower() or "讨论" in ch["title"].lower():
                            expandable_chapters.append(ch["title"])
                    
                    # 如果没有找到合适的章节，则选择主要发现或结论
                    if not expandable_chapters:
                        for ch in chapters:
                            if "主要" in ch["title"] or "结论" in ch["title"]:
                                expandable_chapters.append(ch["title"])
                    
                    # 如果仍然没有找到，使用最后一个非参考文献的章节
                    if not expandable_chapters:
                        for ch in reversed(chapters):
                            if "参考" not in ch["title"].lower():
                                expandable_chapters.append(ch["title"])
                                break
                    
                    # 生成补充内容
                    if expandable_chapters:
                        target_chapter = expandable_chapters[0]
                        words_needed = 12000 - chinese_char_count
                        
                        supplement_prompt = f"""
                        研究主题'{topic}'的报告中，'{target_chapter}'章节需要扩充约{words_needed}字的内容。
                        
                        请提供补充内容，可以包括：
                        1. 更多具体案例或数据
                        2. 不同角度的分析
                        3. 实际应用场景
                        4. 更详细的解释
                        
                        补充内容应与原章节主题一致，并且与研究主题'{topic}'高度相关。
                        请基于以下信息生成约{words_needed}字的高质量补充内容：
                        
                        {optimized_info[:4000]}
                        """
                        
                        try:
                            supplement_content = llm_client.ask([{"role": "user", "content": supplement_prompt}], temperature=0.6, max_tokens=3000)
                            if supplement_content:
                                # 添加补充内容到相应章节
                                for i, part in enumerate(final_report_parts):
                                    if target_chapter in part:
                                        final_report_parts[i] += f"\n\n### 补充分析\n\n{supplement_content}"
                                        break
                                
                                # 重新组合报告
                                final_report = title + subtitle + toc + "\n\n".join(final_report_parts)
                                chinese_char_count = sum(1 for char in final_report if '\u4e00' <= char <= '\u9fff')
                                log(f"添加补充内容后，报告共约{chinese_char_count}中文字")
                        except Exception as e:
                            log(f"生成补充内容时出错: {e}")
                
                completion_narrative = generate_narrative("完成报告", f"研究报告已按学术论文格式分{len(chapters)}章生成完毕", topic)
                log(completion_narrative)
            else:
                final_report = f"(很抱歉，报告生成失败，无法生成任何章节内容。请稍后重试。)"
                error_narrative = generate_narrative("报告生成失败", "无法完成研究报告", topic)
                log(error_narrative)
        except Exception as e:
            log(f"报告生成过程中发生错误: {e}")
            final_report = f"(研究失败: {e})"
        
        if final_report:
            final_report = final_report
            completion_narrative = generate_narrative("完成报告", "研究报告已生成完毕", topic)
            log(completion_narrative)
        else:
            final_report = "(很抱歉，报告生成失败，请稍后重试。)"
            error_narrative = generate_narrative("报告生成失败", "无法完成研究报告", topic)
            log(error_narrative)
            
        # 保存研究数据
        save_research_data(research_id, topic, model_key, logs, final_report, clarification_questions, initial_analysis)
        
        # 添加到历史记录
        add_history_item(research_id, topic, model_key)
        
    except Exception as e:
        log(f"研究过程中发生错误: {e}")
        final_report = f"(研究失败: {e})"
        
        # 即使失败也保存研究数据
        save_research_data(research_id, topic, model_key, logs, final_report, clarification_questions, initial_analysis)
    finally:
        # 停止过滤线程
        if filter_thread and filter_thread.is_alive():
            try:
                log("正在清理过滤线程资源...")
                filter_thread.stop()
                filter_thread.join(2)  # 短等待
            except Exception as e:
                log(f"停止过滤线程时出错: {e}")
                
        is_running = False
        current_stage = "complete"
        return research_id

# 定义HTTP请求处理器
class DeepResearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                error_msg = f"无法加载页面: {e}"
                self.wfile.write(error_msg.encode('utf-8'))
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        elif self.path == '/chatgpt-style' or self.path == '/chatgpt_style.html':
            # 提供ChatGPT风格的界面
            try:
                with open('chatgpt_style.html', 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                error_msg = f"无法加载ChatGPT风格页面: {e}"
                self.wfile.write(error_msg.encode('utf-8'))
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        elif self.path == '/logs':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            status = {
                "logs": logs,
                "finished": (not is_running),
                "stage": current_stage,
                "clarification_questions": clarification_questions
            }
            self.wfile.write(json.dumps(status, ensure_ascii=False).encode('utf-8'))
        elif self.path == '/report':
            self.send_response(200)
            self.send_header('Content-type', 'text/markdown; charset=utf-8')
            self.end_headers()
            self.wfile.write(final_report.encode('utf-8'))
        elif self.path == '/history':
            # 返回研究历史记录列表
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(research_history, ensure_ascii=False).encode('utf-8'))
        elif self.path.startswith('/research/'):
            # 获取特定研究的详细信息
            research_id = self.path.split('/')[-1]
            research_data = load_research_data(research_id)
            
            if research_data:
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(research_data, ensure_ascii=False).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(f"研究ID: {research_id} 未找到".encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        global is_running  # 移动到函数开头，修复语法错误
        if self.path == '/start':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                params = {}
            topic = params.get('topic', '').strip()
            model = params.get('model', '').strip()
            if not topic:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                error = {"error": "研究主题不能为空."}
                self.wfile.write(json.dumps(error, ensure_ascii=False).encode('utf-8'))
            elif is_running:
                self.send_response(429)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                error = {"error": "已有研究任务正在进行，请稍候."}
                self.wfile.write(json.dumps(error, ensure_ascii=False).encode('utf-8'))
            else:
                # 清除旧状态
                global logs, final_report, clarification_questions, clarification_answers, initial_analysis, current_stage
                logs = []
                final_report = ""
                clarification_questions = []
                clarification_answers = {}
                initial_analysis = ""
                current_stage = "initial"
                
                # 生成研究ID
                research_id = str(uuid.uuid4())
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {"status": "started", "research_id": research_id}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
                # 开始澄清阶段
                start_clarification_phase(topic, model, research_id)
                
        elif self.path == '/clarify':
            # 处理用户对澄清问题的回答
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                params = {}
            
            research_id = params.get('research_id', '')
            answers = params.get('answers', {})
            
            if not research_id:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                error = {"error": "缺少研究ID"}
                self.wfile.write(json.dumps(error, ensure_ascii=False).encode('utf-8'))
                return
            
            # 提交回答并继续研究
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            
            # 开始处理回答
            is_running = True
            
            response = {"status": "processing"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            # 在后台线程中处理回答
            processing_thread = threading.Thread(
                target=submit_clarification_answers, 
                args=(research_id, answers)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"Not found")

# 智能检索系统
class SmartRetriever:
    """智能检索系统，用于对文档进行预处理、评分和筛选"""
    
    def __init__(self, embedding_model_key="ollama_qwen3_06b"):
        """初始化检索系统
        Args:
            embedding_model_key: 用于嵌入和文本处理的模型键名
        """
        self.embedding_model_key = embedding_model_key
        # 初始化用于嵌入和处理的轻量级模型
        self.embedding_client = LLMClient(embedding_model_key)
        self.alpha = 0.4  # 关键词密度评分权重
        self.beta = 0.6   # 语义相关度评分权重
        self.chunk_size = 1500  # 文档分块大小（字符数）
        self.chunk_overlap = 200  # 块重叠大小
        
    def chunk_document(self, document):
        """将文档分割成较小的块
        Args:
            document: 完整文档文本
        Returns:
            chunks: 文本块列表
        """
        if not document:
            return []
            
        chunks = []
        i = 0
        doc_len = len(document)
        
        while i < doc_len:
            # 确定块结束位置
            end = min(i + self.chunk_size, doc_len)
            
            # 如果不是最后一块且不在句子边界，尝试找到更好的断点
            if end < doc_len:
                # 尝试在句子结束处断开（句号、问号、感叹号后面接空格或换行）
                better_end = document.rfind('. ', i, end)
                if better_end == -1:
                    better_end = document.rfind('? ', i, end)
                if better_end == -1:
                    better_end = document.rfind('! ', i, end)
                if better_end == -1:
                    better_end = document.rfind('\n', i, end)
                
                # 如果找到更好的断点，就使用它
                if better_end != -1 and better_end > i + self.chunk_size // 3:  # 确保块不会太小
                    end = better_end + 1  # +1 是为了包含句号
            
            # 添加当前块
            chunks.append(document[i:end])
            
            # 移动到下一个起始位置，考虑重叠
            i = end - self.chunk_overlap
            
            # 确保i有进展（防止无限循环）
            if i <= 0:
                i = end
                
        return chunks
    
    def compute_hard_score(self, chunk, keywords):
        """计算关键词密度评分
        Args:
            chunk: 文本块
            keywords: 关键词列表
        Returns:
            normalized_score: 0-1之间的归一化密度得分
        """
        if not chunk or not keywords:
            return 0.0
            
        chunk_lower = chunk.lower()
        total_count = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # 计算关键词出现次数
            count = chunk_lower.count(keyword_lower)
            total_count += count
        
        # 归一化处理（按文本长度）
        chunk_len = max(1, len(chunk) / 100)  # 每100个字符为单位
        normalized_score = min(1.0, total_count / chunk_len)
        
        return normalized_score
    
    def compute_embedding(self, text):
        """计算文本的嵌入向量
        Args:
            text: 输入文本
        Returns:
            embedding: 文本嵌入向量
        """
        global ST_MODEL
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                # 使用全局SentenceTransformer计算嵌入
                # 懒加载模型以节省内存
                if ST_MODEL is None:
                    print("首次加载SentenceTransformer模型...", file=sys.stderr)
                    # 使用小型多语言模型，支持中英文
                    ST_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L6-v2')
                    print("SentenceTransformer模型加载完成", file=sys.stderr)
                
                # 截断长文本以提高性能
                text_to_embed = text[:1000]
                embedding = ST_MODEL.encode(text_to_embed, convert_to_numpy=True)
                return embedding
            except Exception as e:
                print(f"SentenceTransformer嵌入计算出错: {e}，回退到备选方法", file=sys.stderr)
                # 出错时回退到备选方法
        
        # 备选方法：使用轻量级模型生成数值向量
        try:
            prompt = f"仅输出形如0.12,-0.34的10个数字，用逗号隔开：{text[:400]}"
            response = self.embedding_client.ask([{"role": "user", "content": prompt}], temperature=0.1)
            
            # 提取数字部分，处理各种可能的格式
            numbers_str = re.findall(r'-?\d+\.\d+(?=,|$)', response)
            if numbers_str:
                # 转换为浮点数
                embedding = []
                for num_str in numbers_str[:10]:
                    try:
                        embedding.append(float(num_str))
                    except ValueError:
                        embedding.append(0.0)
                
                # 确保正好有10个值
                while len(embedding) < 10:
                    embedding.append(0.0)
                
                return np.array(embedding)
            else:
                # 没有找到数字，返回零向量
                return np.zeros(10)
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}", file=sys.stderr)
            return np.zeros(10)  # 出错时返回零向量
    
    def compute_soft_score(self, chunk, query):
        """计算语义相关度评分
        Args:
            chunk: 文本块
            query: 用户查询/问题
        Returns:
            similarity: 0-1之间的余弦相似度得分
        """
        if not chunk or not query:
            return 0.0
            
        try:
            # 计算查询和文本块的嵌入
            query_embedding = self.compute_embedding(query)
            chunk_embedding = self.compute_embedding(chunk)
            
            # 计算余弦相似度
            dot_product = np.dot(query_embedding, chunk_embedding)
            query_norm = np.linalg.norm(query_embedding)
            chunk_norm = np.linalg.norm(chunk_embedding)
            
            # 防止除零错误
            similarity = 0.0
            if query_norm > 0 and chunk_norm > 0:
                similarity = dot_product / (query_norm * chunk_norm)
                # 确保在0-1范围内
                similarity = max(0.0, min(1.0, similarity))
            
            return similarity
        except Exception as e:
            print(f"计算语义相似度时出错: {e}", file=sys.stderr)
            return 0.0
    
    def extract_keywords(self, query):
        """从查询中提取关键词
        Args:
            query: 用户查询/问题
        Returns:
            keywords: 关键词列表
        """
        try:
            prompt = f"""请从以下查询中提取5-8个关键词（单词或短语），这些关键词对于检索相关文档至关重要。
            
            查询：{query}
            
            只需列出关键词，每行一个，不要有编号或其他多余文字。包括专有名词、术语和主题词。
            """
            
            response = self.embedding_client.ask([{"role": "user", "content": prompt}], temperature=0.3)
            
            # 处理响应，提取关键词
            keywords = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('关键词') and not line.startswith('-') and not line.startswith('*'):
                    # 去除可能的序号
                    clean_line = line
                    # 去除序号：1.，2)，(3)等
                    clean_line = re.sub(r'^[\d\.\)\(\s]+', '', clean_line)
                    clean_line = clean_line.strip()
                    if clean_line:
                        keywords.append(clean_line)
            
            # 添加原始查询中的所有2个字以上的词
            for word in query.split():
                if len(word) >= 2 and word not in keywords:
                    keywords.append(word)
            
            return keywords
        except Exception as e:
            print(f"提取关键词时出错: {e}", file=sys.stderr)
            # 失败时返回简单的空格分词结果
            return [w for w in query.split() if len(w) >= 2]
    
    def filter_and_rank(self, document, query, top_n=5):
        """对文档进行过滤和排序，选出最相关的前N个文本块，并去除重复内容
        Args:
            document: 完整文档文本
            query: 用户查询/问题
            top_n: 返回的最相关块数量
        Returns:
            ranked_chunks: 按相关性排序的前N个文本块
        """
        if not document or not query:
            return []
        
        # 提取关键词
        keywords = self.extract_keywords(query)
        print(f"从查询中提取的关键词: {keywords}", file=sys.stderr)
        
        # 将文档分块
        chunks = self.chunk_document(document)
        print(f"文档分为{len(chunks)}个块", file=sys.stderr)
        
        if not chunks:
            return []
        
        # 计算每个块的得分
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            # 计算关键词密度得分
            hard_score = self.compute_hard_score(chunk, keywords)
            
            # 如果硬得分为0，跳过语义相似度计算（优化性能）
            if hard_score == 0:
                continue
                
            # 计算语义相关度得分
            soft_score = self.compute_soft_score(chunk, query)
            
            # 综合得分
            combined_score = self.alpha * hard_score + self.beta * soft_score
            
            # 存储得分和块
            chunk_scores.append((combined_score, i, chunk))
        
        # 排序并选择得分最高的 top_n*2 个（后面会去重）
        chunk_scores.sort(reverse=True)  # 按得分降序排序
        potential_chunks = chunk_scores[:top_n*2]
        
        # 去重并按原始顺序排序以保持上下文连贯性
        # 使用首句哈希判断内容是否重复
        unique_chunks = []
        first_sentence_hashes = set()
        
        for score, idx, chunk in sorted(potential_chunks, key=lambda x: x[1]):
            # 获取第一句作为判重依据
            first_sentence = chunk.split('.')[0] if '.' in chunk else chunk[:100]
            first_sentence_hash = hash(first_sentence.strip())
            
            # 如果首句哈希不存在，则添加该块
            if first_sentence_hash not in first_sentence_hashes:
                first_sentence_hashes.add(first_sentence_hash)
                unique_chunks.append((score, idx, chunk))
                
                # 一旦收集到足够的唯一块，就停止
                if len(unique_chunks) >= top_n:
                    break
        
        # 按原始顺序排序
        unique_chunks.sort(key=lambda x: x[1])
        
        # 如果去重后不足top_n，调整top_n值
        if len(unique_chunks) < top_n:
            top_n = len(unique_chunks)
        
        # 提取文本块
        ranked_chunks = [chunk for _, _, chunk in unique_chunks]
        
        return ranked_chunks
    
    def process_document(self, document, query):
        """处理文档，返回与查询最相关的部分
        Args:
            document: 完整文档文本
            query: 用户查询/问题
        Returns:
            relevant_text: 与查询最相关的文本
        """
        # 获取排名靠前的块
        top_chunks = self.filter_and_rank(document, query)
        
        if not top_chunks:
            return ""
            
        # 组合最相关的文本块
        relevant_text = "\n\n---\n\n".join(top_chunks)
        
        # 让轻量级模型生成摘要，进一步精简内容
        try:
            summary_prompt = f"""我有一个关于"{query}"的研究问题，从文档中找到了以下可能相关的内容。
            请总结这些内容中与我的问题最相关的要点，保留关键信息和数据，删除不相关内容。
            保持客观，不要添加额外信息，长度控制在原文的70%以内。
            
            文档内容:
            {relevant_text}
            """
            
            summary = self.embedding_client.ask([{"role": "user", "content": summary_prompt}], temperature=0.3, max_tokens=2000)
            
            # 如果摘要生成成功，使用摘要；否则使用原始相关文本
            if summary and len(summary) > 100:
                return summary
            else:
                return relevant_text
                
        except Exception as e:
            print(f"生成摘要时出错: {e}", file=sys.stderr)
            return relevant_text  # 出错时返回原始相关文本
    
    def smart_retrieve(self, collected_info, query):
        """对已收集的信息进行智能检索，过滤出与查询最相关的部分
        Args:
            collected_info: 收集的所有信息文本
            query: 用户查询/主题
        Returns:
            filtered_info: 经过筛选的相关信息
        """
        if not collected_info:
            return ""
            
        print(f"开始智能检索，原始信息长度: {len(collected_info)}字符", file=sys.stderr)
        
        # 处理文档，提取相关部分
        filtered_info = self.process_document(collected_info, query)
        
        print(f"智能检索完成，筛选后信息长度: {len(filtered_info)}字符", file=sys.stderr)
        
        return filtered_info

# 并行文本过滤线程
class TextFilterThread(threading.Thread):
    """使用轻量级模型(qwen3:0.6b)进行独立的文本过滤的线程"""
    
    def __init__(self, topic):
        """初始化过滤线程
        Args:
            topic: 研究主题，用于相关性判断
        """
        super().__init__(daemon=True)  # 设为守护线程，主线程结束时自动结束
        self.topic = topic
        self.running = True
        self.filter_model_key = "ollama_qwen3_06b"  # 专用的轻量级模型
        self.retriever = None
        log("初始化并行文本过滤线程，使用模型：" + self.filter_model_key)
    
    def run(self):
        """线程主循环，不断从队列获取文本并进行过滤"""
        try:
            # 延迟初始化检索系统，避免主线程阻塞
            log("过滤线程正在初始化qwen3:0.6b模型...")
            self.retriever = SmartRetriever(self.filter_model_key)
            log("过滤线程初始化完成，等待文本输入...")
            
            while self.running:
                try:
                    # 从队列中获取待处理文本，超时1秒
                    item = filter_queue.get(timeout=1)
                    if item is None:  # 结束信号
                        break
                    
                    text_id, text, source_info = item
                    
                    # 使用SmartRetriever处理文本
                    try:
                        filtered_text = self.retriever.process_document(text, self.topic)
                        
                        # 保留原始来源信息
                        if source_info:
                            filtered_text = f"{filtered_text}\n\n{source_info}"
                            
                        # 存储过滤结果
                        with filter_lock:
                            filtered_results[text_id] = filtered_text
                            
                        # 记录日志
                        reduction = 100 - (len(filtered_text) * 100 // max(1, len(text)))
                        log(f"并行过滤器处理完成ID:{text_id[:8]}，减少{reduction}%")
                    except Exception as e:
                        log(f"过滤文本时出错: {e}")
                        # 出错时存储原始文本
                        with filter_lock:
                            filtered_results[text_id] = text
                    
                    # 标记任务完成
                    filter_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    log(f"过滤线程处理出错: {e}")
                    continue
        except Exception as e:
            log(f"过滤线程初始化出错: {e}")
        finally:
            log("过滤线程已退出")
    
    def stop(self):
        """停止过滤线程"""
        self.running = False
        # 发送结束信号
        filter_queue.put(None)

# 添加并行过滤文本的函数
def filter_text_parallel(text, topic, source_info=""):
    """将文本提交到并行过滤队列
    Args:
        text: 待过滤文本
        topic: 研究主题
        source_info: 文本来源信息
    Returns:
        text_id: 文本ID，用于后续获取结果
    """
    text_id = str(uuid.uuid4())
    filter_queue.put((text_id, text, source_info))
    return text_id

def get_filtered_text(text_id, timeout=30):
    """获取过滤后的文本
    Args:
        text_id: 文本ID
        timeout: 超时时间(秒)
    Returns:
        filtered_text: 过滤后的文本，或原始文本(超时)
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with filter_lock:
            if text_id in filtered_results:
                return filtered_results.pop(text_id)
        # 等待一小段时间再检查
        time.sleep(0.5)
    
    # 超时返回None
    log(f"获取过滤结果超时: {text_id[:8]}")
    return None

if __name__ == '__main__':
    port = 8000
    
    # 初始化：加载历史记录
    load_history()
    
    print(f"DeepResearch服务器已启动：http://localhost:{port}/")
    server = ThreadingHTTPServer(('', port), DeepResearchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
