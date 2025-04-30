import re
import json
import urllib.request
import urllib.parse
import sys
import time

# 全局LLM客户端（由deepresearch.py初始化）
llm_client = None

# SearXNG 搜索引擎配置（如需可更换为公共实例）
SEARXNG_URL = "http://localhost:8080"  # 如果有可用的公开 SearXNG 实例，请在此修改URL
# 备用公共SearXNG实例
SEARXNG_FALLBACKS = [
    "https://searx.thegpm.org",
    "https://searx.tiekoetter.com",
    "https://searx.be",
    "https://search.sapti.me"
]
SEARXNG_SECRET = "20e218a22baf64179791559447f3992b65d5e5bbac8980970c9b1"  # 如果需要API密钥，请在此填入

def init_llm_client(client):
    """初始化搜索模块使用的 LLM 客户端"""
    global llm_client
    llm_client = client

def clean_text(text: str) -> str:
    """清理提取的文本，去除多余的空白和控制字符"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text.strip()

def generate_query(query: str) -> str:
    """利用LLM生成针对用户查询的多个搜索关键字变体"""
    prompt = (
        "您是一位专业的研究助手。根据用户的查询，请生成最多四个不同的、精确的搜索查询，以帮助收集有关该主题的全面信息。\n"
        "请只返回一个Python列表格式的字符串，例如：['查询1', '查询2', '查询3']\n"
        "注意：必须使用英文引号和方括号，确保返回的是有效的Python语法。不要添加任何其他解释或内容。"
    )
    messages = [
        {"role": "system", "content": "您是一位专业、精确的研究助手。请严格按照要求的格式返回结果。"},
        {"role": "user", "content": f"用户查询: {query}\n\n{prompt}"}
    ]
    if not llm_client:
        raise Exception("LLM client not initialized.")
    response = llm_client.ask(messages, temperature=0.7, max_tokens=512)
    return response

def if_useful(query: str, page_text: str) -> str:
    """使用LLM判断页面内容是否与查询相关、有用"""
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, determine if the webpage contains information relevant and useful for addressing the query.\n"
        "Please provide a brief explanation of your evaluation, followed by your decision.\nEnd with a clear 'Yes' if the page is useful, or 'No' if it is not."
    )
    content_snippet = page_text[:20000]  # 限制内容长度以防止超出上下文
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {query}\n\nWebpage Content (snippet):\n{content_snippet}\n\n{prompt}"}
    ]
    response = llm_client.ask(messages, temperature=0.0, max_tokens=256)
    if not response:
        return "No"
    lines = response.strip().split('\n')
    decision_line = lines[-1].strip()
    if "Yes" in decision_line:
        return "Yes"
    elif "No" in decision_line:
        return "No"
    return "No"

def extract_relevant_context(query: str, search_query: str, page_text: str) -> str:
    """使用LLM从页面内容中提取与查询相关的信息片段"""
    content = page_text
    if len(content) > 20000:
        content = content[:20000]
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, and the webpage content, extract all pieces of information that are relevant to answering the user's query.\n\n"
        "Your extraction should:\n"
        "1. Focus on factual, relevant information directly related to the query\n"
        "2. Maintain the original wording where possible\n"
        "3. Preserve important context, dates, statistics, and quotes\n"
        "4. Organize information coherently\n\n"
        "Return the extracted information in a well-structured format."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {query}\nSearch Query: {search_query}\n\nWebpage Content:\n{content}\n\n{prompt}"}
    ]
    response = llm_client.ask(messages, temperature=0.0, max_tokens=1024)
    if response:
        return response.strip()
    return ""

def get_new_search_queries(user_query: str, previous_search_queries, all_contexts) -> list:
    """利用LLM分析是否需要进一步搜索，并生成新搜索查询"""
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, and the extracted contexts from webpages, determine if further research is needed.\n\n"
        "First, analyze what information we already have and what important aspects are still missing.\n\n"
        "Then, if further research is needed, provide up to four new search queries as a Python list (for example, ['new query1', 'new query2']). These queries should target specific information gaps or explore areas not yet covered.\n\n"
        "If you believe no further research is needed because we have comprehensive information, respond with an empty list: []\n\n"
        "Start with your analysis, then provide your decision on new queries."
    )
    messages = [
        {"role": "system", "content": "You are an expert in research strategy and information analysis."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = llm_client.ask(messages, temperature=0.7, max_tokens=512)
    if not response:
        return []
    match = re.search(r'\[(.*?)\]', response)
    if match:
        list_str = match.group(0)
        try:
            new_queries = eval(list_str)
            if isinstance(new_queries, list):
                return new_queries
        except Exception:
            pass
    # 检查响应中是否表明不需要进一步搜索
    if "no further research" in response.lower() or "无需进一步搜索" in response:
        return []
    return []

def try_searxng_instance(base_url, query, headers, params, top_k, categories, timeout=30):
    """尝试使用特定的SearXNG实例进行搜索"""
    full_url = f"{base_url}/search?{urllib.parse.urlencode(params)}"
    links = []
    
    try:
        print(f"尝试使用SearXNG实例: {base_url}", file=sys.stderr)
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            try:
                results = json.loads(data.decode('utf-8')).get('results', [])
                for result in results[:top_k]:
                    if categories == 'images':
                        link = result.get('img_src')
                    else:
                        link = result.get('url')
                    if link:
                        links.append(link)
                return links, True  # 成功
            except json.JSONDecodeError as e:
                print(f"解析搜索结果失败: {e}", file=sys.stderr)
                return [], False  # 失败
    except Exception as e:
        print(f"搜索实例 {base_url} 请求失败: {e}", file=sys.stderr)
        return [], False  # 失败

def web_search(query: str, top_k: int = 3, categories: str = 'general', max_retries: int = 3) -> list:
    """调用SearXNG搜索API，返回结果链接列表（images类别返回图片URL列表）"""
    params = {
        'q': query,
        'format': 'json',
        'language': 'zh-CN',
        'safesearch': '0',
        'categories': categories
    }
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    if SEARXNG_SECRET:
        headers['X-API-Key'] = SEARXNG_SECRET
    
    # 首先尝试本地实例
    links, success = try_searxng_instance(SEARXNG_URL, query, headers, params, top_k, categories)
    if success and links:
        return links
    
    print(f"本地SearXNG实例失败或无结果，尝试公共备用实例", file=sys.stderr)
    
    # 本地实例失败，尝试公共备用实例
    for fallback_url in SEARXNG_FALLBACKS:
        links, success = try_searxng_instance(fallback_url, query, headers, params, top_k, categories)
        if success and links:
            print(f"使用备用实例 {fallback_url} 成功", file=sys.stderr)
            return links
        # 每次失败后等待一小段时间，避免过于频繁请求
        time.sleep(1)
    
    # 如果所有实例都失败，进行多次重试
    print(f"所有SearXNG实例均失败，开始重试...", file=sys.stderr)
    
    # 组合所有可能的URL
    all_urls = [SEARXNG_URL] + SEARXNG_FALLBACKS
    
    # 重试机制
    for retry in range(1, max_retries):
        print(f"重试第 {retry} 次 (共 {max_retries} 次)", file=sys.stderr)
        
        # 每次重试使用不同的URL
        for url in all_urls:
            links, success = try_searxng_instance(url, query, headers, params, top_k, categories, timeout=30 + retry * 10)
            if success and links:
                return links
            time.sleep(2)  # 短暂等待
    
    print(f"所有SearXNG实例在多次重试后仍然失败", file=sys.stderr)
    return []  # 返回空列表

def fetch_webpage_text(url: str, max_retries: int = 2) -> str:
    """获取网页内容并返回纯文本（去除HTML标签和脚本样式）"""
    # 重试机制
    for retry in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            # 增加超时时间到45秒
            with urllib.request.urlopen(req, timeout=45) as resp:
                content_type = resp.getheader('Content-Type', '')
                content_bytes = resp.read()
                encoding = 'utf-8'
                if 'charset=' in content_type:
                    charset = content_type.split('charset=')[-1]
                    encoding = charset
                html_content = content_bytes.decode(encoding, errors='ignore')
                
                # 直接获取成功，跳过Jina代理
                break
        except Exception as e:
            print(f"直接获取网页失败 (尝试 {retry+1}/{max_retries}): {str(e)}", file=sys.stderr)
            if retry < max_retries - 1:
                # 如果不是最后一次重试，等待后继续
                time.sleep(2)
                continue
                
            # 直接获取失败且重试用尽，尝试通过Jina代理
            try:
                proxy_url = f"https://r.jina.ai/{url}"
                # Jina代理增加超时时间到60秒
                with urllib.request.urlopen(proxy_url, timeout=60) as resp:
                    html_content = resp.read().decode('utf-8', errors='ignore')
                    break  # 成功获取数据，跳出循环
            except Exception as proxy_e:
                print(f"通过Jina代理获取失败: {str(proxy_e)}", file=sys.stderr)
                return ""  # 所有方法尝试失败
    else:
        # 如果循环正常结束但未break，说明所有尝试均失败
        return ""
    
    # 成功获取内容，处理HTML
    try:
        # 移除不需要的元素（脚本、样式等）
        html_content = re.sub(r'(?is)<(script|style|title|head|meta|noscript|header|footer|aside|nav)[^>]*>.*?</\1>', ' ', html_content)
        # 移除所有HTML标签
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        # 反转义HTML实体
        try:
            import html
            text_content = html.unescape(text_content)
        except ImportError:
            pass
        # 清理文本
        text_content = clean_text(text_content)
        return text_content
    except Exception as parse_e:
        print(f"解析HTML内容失败: {str(parse_e)}", file=sys.stderr)
        return ""

def process_link(link: str, query: str, search_query: str):
    """处理单个链接：获取网页文本，判断有用性，提取相关内容"""
    page_text = fetch_webpage_text(link)
    if not page_text:
        return None
    usefulness = if_useful(query, page_text)
    if usefulness == "Yes":
        context = extract_relevant_context(query, search_query, page_text)
        if context:
            domain = urllib.parse.urlparse(link).netloc
            context_with_source = f"{context}\n\n来源: {domain} ({link})"
            return context_with_source
    return None

def get_images_description(image_url: str) -> str:
    """使用LLM对图片进行简单描述（如果模型不支持图像，可返回占位描述）"""
    prompt = (
        f"请看这张图片：{image_url}\n"
        "请使用一句话详细描述这张图片的内容，包括可见的重要特征。"
    )
    # 注意：多数LLM无法直接解析图片，这里简单返回占位描述
    return "与主题相关的图片"

def generate_narrative(action: str, context: str = None, topic: str = None) -> str:
    """让LLM生成自然的第一人称叙述，描述当前的操作和发现"""
    if not llm_client:
        # 如果没有LLM客户端，返回基本描述
        return f"正在{action}..."
    
    prompt = (
        f"作为一个专业、高效的研究助手，请用简洁专业的第一人称描述你正在进行的以下操作:\n"
        f"操作: {action}\n"
        f"主题: {topic if topic else '未指定'}\n"
        f"上下文: {context if context else '无'}\n\n"
        f"你的描述应该平衡专业性和亲和力，避免过于幼稚或过于情绪化的表达。"
        f"描述应该简洁（50-100字），同时信息丰富。直接给出描述，不要有多余的引言。"
        f"示例风格: '我正在分析X主题的关键数据，发现了Y趋势，这可能与Z因素相关。'"
    )
    
    messages = [
        {"role": "system", "content": "你是一个专业、高效的研究助手，使用平衡的语言描述你的工作过程。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_client.ask(messages, temperature=0.7, max_tokens=100)
        return response.strip()
    except Exception:
        # 如果生成失败，返回基本描述
        return f"正在{action}..."

def get_images(query: str):
    """搜索与主题相关的图片URL（及描述）"""
    image_urls = web_search(query, top_k=3, categories='images')
    if not image_urls:
        return "未找到相关图片"
    result = {}
    for img_url in image_urls:
        try:
            description = get_images_description(img_url)
        except Exception:
            description = ""
        result[img_url] = description if description else ""
    return result
