# ================================================
# OpenRouter Reverse Proxy - 生产部署版本 (Render)
# 支持超长 Conversation History + 自动总结 + 返回日志
# 新增: 浏览器访问 /logs 可实时查看日志 + 自动总结状态
# 新增: Google Docs 实时更新角色知识库（无需重新部署）
# ================================================

import json
import requests
import time
from datetime import datetime
from functools import lru_cache
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================== 配置参数 ==================

model = "openrouter/auto"
auto_trim = True

# ================== 长上下文 + 总结配置 ==================

MAX_CONTEXT_TOKENS = 200000
KEEP_RECENT_TOKENS = 18000
SUMMARY_EVERY_TOKENS = 80000

# Advance settings
min_p = 0.04
top_p = 0.95
top_k = 80
repetition_penalty = 1.10
frequency_penalty = 0.05
presence_penalty = 0.08

prefill_enabled = False
assistant_prefill = "..."  

# ================== 外部角色知识库（Google Docs） ==================
# 【重要】请把下面这行改成你自己的 Google Docs 发布链接
GOOGLE_DOC_PUB_URL = "https://docs.google.com/document/d/你的文档ID/pub"  

@lru_cache(maxsize=1)
def get_character_knowledge():
    """从 Google Docs 获取最新角色资料，每5分钟自动更新一次"""
    try:
        url = GOOGLE_DOC_PUB_URL.rstrip('/') + "?embedded=true"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        
        # 清理多余空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        print(f"[INFO] 成功从 Google Docs 加载知识库，长度: {len(clean_text)} 字符")
        return clean_text
    except Exception as e:
        print(f"[ERROR] 从 Google Docs 加载知识库失败: {str(e)}")
        return "【知识库加载失败，请检查 Google Doc 链接是否正确且已发布为公开】"

# ================== 全局日志存储和总结记录 ==================
logs = []                    # 存储所有日志行 (最多保留2000行，防止内存过大)
last_summary = None          # 最近一次自动总结的内容
last_summary_time = None     # 总结发生的时间

def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    logs.append(log_line)
    if len(logs) > 2000:
        logs.pop(0)
    print(log_line)

def add_summary(summary_content):
    global last_summary, last_summary_time
    last_summary = summary_content
    last_summary_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_log(f"自动总结已触发 | 时间: {last_summary_time} | 总结长度: {len(summary_content)} 字符")

# ================== 日志函数 ===================

def log_info(message):
    add_log(f"[INFO] {message}")

def log_response(content, model_name="Unknown", is_stream=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"\n{'='*90}\n[RESPONSE LOG {timestamp}] Model: {model_name} | Stream: {is_stream}\n[LENGTH] {len(content)} characters\n[CONTENT START]\n{content}\n[CONTENT END]\n{'='*90}\n"
    add_log(log_line)

# ================== 辅助函数 ==================

def trim_to_end_sentence(input_str, include_newline=False):
    punctuation = set(['.', '!', '?', '*', '"', ')', '}', '`', ']', '$', '。', '！', '？', '”', '）', '】', '’', '」'])
    last = -1
    for i in range(len(input_str) - 1, -1, -1):
        char = input_str[i]
        if char in punctuation:
            last = i - 1 if i > 0 and input_str[i - 1] in [' ', '\n'] else i
            break
        if include_newline and char == '\n':
            last = i
            break
    return input_str[:last + 1].rstrip() if last != -1 else input_str.rstrip()

def autoTrim(text):
    return trim_to_end_sentence(text)

def estimate_tokens(messages):
    return sum(len(str(msg.get("content", ""))) // 4 + 20 for msg in messages)

# ================== 永久知识库注入（使用 Google Docs） ==================
def ensure_permanent_knowledge(messages):
    if not messages:
        messages = []
    
    kb_instruction = f"""【永久角色知识库 - LLM必须主动检索并严格遵守】
{get_character_knowledge()}

【强制指令】
在生成每次回复前，你必须首先主动检索以上完整知识库的内容，严格根据角色的行为，性格，目的来回应。
绝不允许忘记、淡化或偏离这些设定。这就是你作为角色的长期稳定核心记忆，无论对话历史多长、如何压缩，都必须100%保持一致性。
"""

    for msg in messages:
        if msg.get("role") == "system":
            if "永久角色知识库" not in msg.get("content", ""):
                msg["content"] = msg.get("content", "").strip() + "\n\n" + kb_instruction
            return messages
    
    messages.insert(0, {"role": "system", "content": kb_instruction})
    return messages

# ================== 自动总结函数 ==================
def summarize_old_messages(old_messages):
    if len(old_messages) < 3:
        return None

    summary_prompt = {  
        "role": "system",  
        "content": "You are a professional conversation summarizer. Summarize the following chat history into a concise, coherent memory. Focus on key events, character conversations, important facts, personality traits shown, and story progress. Write in third person. Do not add new information."  
    }  
     
    user_content = "Summarize this conversation history:\n\n" + "\n".join(  
        f"{msg['role']}: {msg['content']}" for msg in old_messages  
    )  
     
    try:  
        api_key = request.headers.get('Authorization', '').strip()  
        summary_response = requests.post(  
            "https://openrouter.ai/api/v1/chat/completions",  
            headers={  
                "Content-Type": "application/json",  
                "Authorization": api_key,  
                "HTTP-Referer": "https://janitorai.com/",  
            },  
            json={  
                "model": "deepseek/deepseek-chat",  
                "messages": [summary_prompt, {"role": "user", "content": user_content}],  
                "max_tokens": 800,  
                "temperature": 0.7  
            }  
        )  
         
        if summary_response.status_code == 200:  
            summary_text = summary_response.json()["choices"][0]["message"]["content"]
            full_summary = f"[MEMORY SUMMARY]\n{summary_text}\n\n[Continue the story from the latest messages]"
            add_summary(full_summary)
            return {"role": "system", "content": full_summary}  
    except Exception as e:
        log_info(f"总结失败: {str(e)}")
    return None

# ================== 历史处理主函数 ==================
def compress_history(messages):
    if len(messages) <= 6:
        return messages

    total_tokens = estimate_tokens(messages)  
     
    if total_tokens <= MAX_CONTEXT_TOKENS:  
        return messages  
     
    log_info(f"History too long: \~{total_tokens} tokens -> auto summarizing...")  
     
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None  
    chat_messages = messages[1:] if system_msg else messages  
     
    recent_messages = []  
    current_tokens = 0  
    for msg in reversed(chat_messages):  
        msg_tokens = len(str(msg.get("content", ""))) // 4 + 20  
        if current_tokens + msg_tokens > KEEP_RECENT_TOKENS:  
            break  
        recent_messages.append(msg)  
        current_tokens += msg_tokens  
    recent_messages.reverse()  
     
    old_messages = chat_messages[:-len(recent_messages)] if len(recent_messages) < len(chat_messages) else []  
    summary = summarize_old_messages(old_messages) if old_messages else None  
     
    new_messages = []  
    if system_msg:  
        new_messages.append(system_msg)  
    if summary:  
        new_messages.append(summary)  
    new_messages.extend(recent_messages)  
     
    log_info(f"Compressed with summary: {len(messages)} -> {len(new_messages)} messages")
    return new_messages

# Stream 处理
def genstream(config, model_name):
    full_content = ""
    try:
        with requests.post(**config) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    text = line.decode('utf-8')
                    if text != ": OPENROUTER PROCESSING":
                        if text.startswith("data: ") and text != "data: [DONE]":
                            try:
                                chunk = json.loads(text[6:])
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_content += delta
                            except:
                                pass
                        yield f"{text}\n\n"
                    time.sleep(0.02)
    except Exception as e:
        log_info(f"Stream error: {e}")
    finally:
        if full_content:
            log_response(full_content, model_name, is_stream=True)

# ================== 浏览器日志页面 ==================
LOG_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Proxy 日志监控</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
        pre { background: #252526; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-break: break-all; }
        .summary { background: #2d2d2d; padding: 15px; margin: 15px 0; border-left: 4px solid #007acc; }
        .log-container { max-height: 80vh; overflow-y: auto; }
        h1 { color: #569cd6; }
    </style>
</head>
<body>
    <h1>Proxy 实时日志监控</h1>
    <p>最后更新: <span id="time"></span> | <a href="#" onclick="location.reload()">刷新</a></p>
    
    <h2>最近自动总结状态</h2>
    <div class="summary">
        {% if last_summary %}
            <strong>总结时间:</strong> {{ last_summary_time }}<br>
            <strong>总结内容:</strong><br>
            <pre>{{ last_summary }}</pre>
        {% else %}
            <em>尚未触发自动总结</em>
        {% endif %}
    </div>

    <h2>完整日志 (实时更新)</h2>
    <div class="log-container">
        <pre id="logs">{{ logs }}</pre>
    </div>

    <script>
        const eventSource = new EventSource('/logs/stream');
        eventSource.onmessage = function(e) {
            const logsDiv = document.getElementById('logs');
            logsDiv.textContent += e.data + '\\n';
            logsDiv.scrollTop = logsDiv.scrollHeight;
            if (Math.random() < 0.1) {
                document.getElementById('time').textContent = new Date().toLocaleString('zh-CN');
            }
        };
    </script>
</body>
</html>
"""

@app.route('/logs')
def show_logs():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(LOG_PAGE_HTML, 
                                  logs='\n'.join(logs[-300:]),
                                  last_summary=last_summary,
                                  last_summary_time=last_summary_time,
                                  current_time=current_time)

@app.route('/logs/stream')
def log_stream():
    def generate():
        last_index = len(logs)
        while True:
            if len(logs) > last_index:
                for line in logs[last_index:]:
                    yield f"data: {line}\n\n"
                last_index = len(logs)
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

# ================== 主处理函数 ==================
def normalOperation(req):
    if not req.json:
        return jsonify(error=True), 400

    data = req.json.copy()  
    if "stream" not in data:  
        data['stream'] = False  

    log_info(f"New request received | Stream: {data.get('stream')} | Model: {data.get('model')}")  

    if "messages" in data:
        data["messages"] = ensure_permanent_knowledge(data["messages"])

    if "messages" in data:  
        data["messages"] = compress_history(data["messages"])  

    if prefill_enabled and data.get("messages"):
        messages = data["messages"]
        if messages[-1]["role"] == "user":
            messages.append({"content": assistant_prefill, "role": "assistant"})
        else:
            messages[-1]["content"] += "\n" + assistant_prefill

    api_url = 'https://openrouter.ai/api/v1'  
    api_key = req.headers.get('Authorization', '').strip()  

    if not api_key:  
        log_info("Error: No API key provided")  
        return jsonify(error="No API key"), 401  

    req_model = data.get("model")  
    newmodel = None if req_model in ["openrouter/auto", "auto", None] else req_model  

    config = {  
        'url': f'{api_url}/chat/completions',  
        'headers': {  
            'Content-Type': 'application/json',  
            'Authorization': api_key,  
            'HTTP-Referer': 'https://janitorai.com/',  
        },  
        'json': {  
            'messages': data.get('messages'),  
            'model': newmodel,  
            'temperature': data.get('temperature', 0.85),  
            'max_tokens': data.get('max_tokens', 4096),  
            'stream': data.get('stream', False),  
            'repetition_penalty': data.get('repetition_penalty', repetition_penalty),  
            'presence_penalty': data.get('presence_penalty', presence_penalty),  
            'frequency_penalty': data.get('frequency_penalty', frequency_penalty),  
            'min_p': data.get('min_p', min_p),  
            'top_p': data.get('top_p', top_p),  
            'top_k': data.get('top_k', top_k),  
            'stop': data.get('stop'),  
            'logit_bias': data.get('logit_bias', {}),  
            'transforms': ["middle-out"],  
        },  
    }  

    try:  
        if data.get('stream'):  
            return Response(stream_with_context(genstream(config, req_model)),   
                          content_type='text/event-stream')  
        else:  
            response = requests.post(**config)  
            if response.status_code <= 299:  
                result = response.json()  
                if result.get("choices") and result["choices"][0].get("message"):  
                    content = result["choices"][0]["message"]["content"]  
                    log_response(content, req_model, is_stream=False)  
                    if auto_trim:  
                        result["choices"][0]["message"]["content"] = autoTrim(content)  
                return jsonify(result)  
            else:  
                log_info(f"API Error {response.status_code}: {response.text}")  
                return jsonify(error=response.json()), response.status_code  
    except Exception as e:  
        log_info(f"Exception occurred: {str(e)}")  
        return jsonify(error=str(e)), 500

@app.route('/')
def default():
    return {"status": "online", "model": model}

@app.route('/models')
@app.route('/v1/models')
def modelcheck():
    return {"object": "list", "data": [{"id": model, "object": "model", "created": 1685474247, "owned_by": "openai"}]}

@app.route("/", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def generate():
    return normalOperation(request)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)