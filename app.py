# ================================================
# OpenRouter Reverse Proxy - Stream + Tool Calling 最终修复版 (DeepSeek-V3.2)
# 专为 anime.gf 设计：Stream 模式下强制 Tool Calling + 修复 UI 重复句子问题
# 核心修复：当检测到 tool call 时，立即停止当前流、执行工具、重新发起新请求生成最终回复
# 避免重复句子 + 保证前端流式显示正常
# ================================================
import json
import requests
import time
from datetime import datetime, timezone, timedelta
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

# Advance settings
min_p = 0.04
top_p = 0.95
top_k = 80
repetition_penalty = 1.10
frequency_penalty = 0.05
presence_penalty = 0.08

# ================== Google Docs 知识库 ==================
# 【必须修改】改成你自己的 /pub 链接
GOOGLE_DOC_PUB_URL = "https://docs.google.com/document/d/e/2PACX-1vSzjLiOsCGRuhn_vlnhSsUMoW1ZYqcj-YmlvKmhCC22Q_w_JAYL3xyDr2FeKBnmtsEObAEH7kx_fipv/pub"

last_google_doc_load_time = None
_character_knowledge_cache = None
_last_knowledge_load_time = None

def get_character_knowledge():
    global last_google_doc_load_time, _character_knowledge_cache, _last_knowledge_load_time
    now = datetime.now(timezone(timedelta(hours=8)))
    
    if (_character_knowledge_cache is not None and 
        _last_knowledge_load_time is not None and
        (now - _last_knowledge_load_time).total_seconds() < 300):
        return _character_knowledge_cache
    
    try:
        url = GOOGLE_DOC_PUB_URL.rstrip('/') + "?embedded=true"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        
        _character_knowledge_cache = clean_text
        _last_knowledge_load_time = now
        sg_time = now.strftime("%Y-%m-%d %H:%M")
        last_google_doc_load_time = sg_time
        print(f"[INFO] Google Docs 知识库加载成功，长度: {len(clean_text)} 字符")
        return clean_text
    except Exception as e:
        print(f"[ERROR] Google Docs 加载失败: {str(e)}")
        if _character_knowledge_cache:
            return _character_knowledge_cache
        return "【知识库加载失败，请检查 /pub 链接】"

# ================== 全局日志 ==================
logs = []
last_summary = None
last_summary_time = None

def add_log(message):
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    logs.append(log_line)
    if len(logs) > 100:
        logs.pop(0)
    print(log_line)

def log_info(message):
    add_log(f"[INFO] {message}")

def log_response(content, model_name="Unknown", is_stream=True):
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"\n{'='*90}\n[RESPONSE LOG {timestamp}] Model: {model_name} | Stream: {is_stream}\n[LENGTH] {len(content)} characters\n[CONTENT START]\n{content}\n[CONTENT END]\n{'='*90}\n"
    add_log(log_line)

# ================== Tool 定义 ==================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_character_knowledge",
            "description": "当你需要查阅角色的详细设定、性格、背景、行为规则或长期记忆时，必须调用此工具。我会返回最新的 Google Docs 完整内容。先调用工具，再生成回复。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

def execute_tool(tool_call):
    kb_text = get_character_knowledge()
    load_time = last_google_doc_load_time or "N/A"
    kb_len = len(kb_text)
    log_info(f"[KB TOOL RESPONSE] 已拉取最新知识库并返回给 LLM | 长度: {kb_len} 字符 | 加载时间: {load_time}")
    return kb_text

# ================== System Prompt ==================
SYSTEM_PROMPT = """你是一个角色扮演助手。你有一个外部知识库工具：retrieve_character_knowledge。
当对话需要角色设定、性格、背景或长期记忆时，你必须先调用工具获取最新信息，再生成回复。"""

# ================== Legacy Injection（备用） ==================
def legacy_ensure_permanent_knowledge(messages):
    if not messages:
        messages = []
    kb_text = get_character_knowledge()
    kb_instruction = f"""【永久角色知识库 - LLM必须主动检索并严格遵守】
{kb_text}
【强制指令】在生成每次回复前，必须首先主动检索以上完整知识库的内容。"""
    marker = "【永久角色知识库"
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if marker in content:
                parts = content.split(marker, 1)
                msg["content"] = parts[0].strip() + "\n\n" + kb_instruction
            else:
                msg["content"] = content.strip() + "\n\n" + kb_instruction
            return messages
    messages.insert(0, {"role": "system", "content": kb_instruction})
    log_info(f"[KB CONSULT] Legacy Injection 已注入知识库 | 长度: {len(kb_text)} 字符")
    return messages

# ================== 辅助函数 ==================
def trim_to_end_sentence(input_str):
    punctuation = set(['.', '!', '?', '*', '"', ')', '}', '`', ']', '$', '。', '！', '？', '”', '）', '】', '’', '」'])
    for i in range(len(input_str)-1, -1, -1):
        if input_str[i] in punctuation:
            return input_str[:i+1].rstrip()
    return input_str.rstrip()

def autoTrim(text):
    return trim_to_end_sentence(text)

def estimate_tokens(messages):
    return sum(len(str(msg.get("content", ""))) // 4 + 20 for msg in messages)

def add_summary(summary_content):
    global last_summary, last_summary_time
    last_summary = summary_content
    last_summary_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    add_log(f"自动总结已触发 | 时间: {last_summary_time}")

def summarize_old_messages(old_messages):
    if len(old_messages) < 3:
        return None
    summary_prompt = {"role": "system", "content": "You are a professional conversation summarizer..."}
    user_content = "Summarize this conversation history:\n\n" + "\n".join(f"{msg['role']}: {msg['content']}" for msg in old_messages)
    try:
        api_key = request.headers.get('Authorization', '').strip()
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Content-Type": "application/json", "Authorization": api_key, "HTTP-Referer": "https://janitorai.com/"},
                          json={"model": "deepseek/deepseek-chat", "messages": [summary_prompt, {"role": "user", "content": user_content}], "max_tokens": 800, "temperature": 0.7})
        if r.status_code == 200:
            summary_text = r.json()["choices"][0]["message"]["content"]
            full_summary = f"[MEMORY SUMMARY]\n{summary_text}\n\n[Continue the story from the latest messages]"
            add_summary(full_summary)
            return {"role": "system", "content": full_summary}
    except:
        pass
    return None

def compress_history(messages):
    if len(messages) <= 6:
        return messages
    total_tokens = estimate_tokens(messages)
    if total_tokens <= MAX_CONTEXT_TOKENS:
        return messages
    log_info(f"History too long: ~{total_tokens} tokens -> auto summarizing...")
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
    if system_msg: new_messages.append(system_msg)
    if summary: new_messages.append(summary)
    new_messages.extend(recent_messages)
    log_info(f"Compressed: {len(messages)} -> {len(new_messages)} messages")
    return new_messages

def log_full_prompt(messages, mode):
    prompt_str = json.dumps(messages, ensure_ascii=False, indent=2)
    add_log(f"[PROMPT LOG] 处理模式: {mode}\n发出的完整 Prompt:\n{prompt_str}\n[PROMPT END]")

# ================== Stream + Tool Calling 核心修复版 ==================
def genstream_with_tool(config, model_name, original_messages):
    """Stream 模式下支持 Tool Calling，修复 UI 重复句子问题"""
    messages = original_messages.copy()
    max_tool_loops = 3
    loop = 0

    while loop < max_tool_loops:
        full_content = ""
        tool_calls_detected = None

        try:
            with requests.post(**config) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    text = line.decode('utf-8')
                    if text == ": OPENROUTER PROCESSING":
                        yield f"{text}\n\n"
                        continue
                    if text.startswith("data: "):
                        if text == "data: [DONE]":
                            yield f"{text}\n\n"
                            continue
                        try:
                            chunk = json.loads(text[6:])
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            # 正常内容流式输出
                            if delta.get("content"):
                                content_delta = delta["content"]
                                full_content += content_delta
                                yield f"{text}\n\n"

                            # 检测 tool call
                            if delta.get("tool_calls"):
                                if tool_calls_detected is None:
                                    tool_calls_detected = []
                                for tc_delta in delta["tool_calls"]:
                                    tool_calls_detected.append(tc_delta)

                            # 工具调用完成
                            if finish_reason == "tool_calls" and tool_calls_detected:
                                log_info(f"[KB TOOL CALL in Stream] Loop {loop+1} 检测到 Tool Call，正在执行...")
                                # 执行工具
                                for tc in tool_calls_detected:
                                    tool_result = execute_tool(tc)
                                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                                    messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": tool_result})
                                
                                # 更新 config，进入下一轮（最终回复）
                                config['json']["messages"] = messages
                                loop += 1
                                break  # 停止当前流，进入下一轮新请求

                        except Exception as parse_e:
                            log_info(f"Stream chunk parse error: {str(parse_e)}")
                    else:
                        yield f"{text}\n\n"
                    time.sleep(0.01)

        except Exception as e:
            log_info(f"Stream error: {e}")
            break

        if not tool_calls_detected:
            # 本轮没有 tool call → 最终回复完成
            if full_content:
                log_response(full_content, model_name, is_stream=True)
            break

    # 如果超过循环次数，强制结束
    if loop >= max_tool_loops:
        log_info("[WARNING] Tool loop 达到上限")

# ================== 日志页面 ==================
LOG_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Proxy 日志监控</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
        pre { background: #252526; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-break: break-all; }
        .info { background: #2d2d2d; padding: 12px; margin: 10px 0; border-left: 4px solid #28a745; }
        .log-container { max-height: 80vh; overflow-y: auto; }
    </style>
</head>
<body>
    <h1>Proxy 实时日志监控</h1>
    <div class="info">
        <strong>Google Docs 最后读取时间:</strong> {{ last_google_doc_time }}
    </div>
    <h2>完整日志 (最新 100 条)</h2>
    <div class="log-container">
        <pre id="logs">{{ logs }}</pre>
    </div>
    <script>
        const eventSource = new EventSource('/logs/stream');
        eventSource.onmessage = function(e) {
            const logsDiv = document.getElementById('logs');
            logsDiv.textContent += e.data + '\\n';
            logsDiv.scrollTop = logsDiv.scrollHeight;
        };
    </script>
</body>
</html>"""

@app.route('/logs')
def show_logs():
    last_google_time = last_google_doc_load_time or "尚未读取"
    return render_template_string(LOG_PAGE_HTML, logs='\n'.join(logs), last_google_doc_time=last_google_time)

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
        data['stream'] = True   # 强制保持 stream（anime.gf 需要）
    
    log_info(f"New request received | Stream: {data.get('stream')} | Model: {data.get('model')}")
    
    messages = data.get("messages", [])
    
    if not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    messages = compress_history(messages)
    
    # ================== 强制 Tool Calling 模式 ==================
    mode = "Tool Calling (Stream 强制模式)"
    log_info(f"[MODE] {mode}")
    data["messages"] = messages
    data["tools"] = TOOLS
    data["tool_choice"] = "auto"
    
    # 打印完整 Prompt（便于调试）
    log_full_prompt(data["messages"], mode)
    
    api_url = 'https://openrouter.ai/api/v1'
    api_key = req.headers.get('Authorization', '').strip()
    if not api_key:
        return jsonify(error="No API key"), 401
    
    req_model = data.get("model")
    newmodel = None if req_model in ["openrouter/auto", "auto", None] else req_model
    
    config = {
        'url': f'{api_url}/chat/completions',
        'headers': {'Content-Type': 'application/json', 'Authorization': api_key, 'HTTP-Referer': 'https://janitorai.com/'},
        'json': data
    }
    
    try:
        # Stream 模式下使用修复后的 Tool Calling 生成器
        return Response(stream_with_context(genstream_with_tool(config, req_model, messages)), 
                        content_type='text/event-stream')
    
    except Exception as e:
        log_info(f"Exception occurred: {str(e)}")
        return jsonify(error=str(e)), 500

# ================== 路由 ==================
@app.route('/')
def default():
    return {"status": "online", "mode": "Stream Forced Tool Calling (UI Repeat Fixed)"}

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
