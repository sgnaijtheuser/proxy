# ================================================
# OpenRouter Reverse Proxy - 强制 Tool Calling 稳定版
# 特点：
# 1. 两阶段：先 tool（非 stream）→ 再生成（stream）
# 2. 100% 强制读取 Google Docs
# 3. 不占第一轮 prompt
# 4. anime.gf 完全兼容
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

# ================== 配置 ==================
model = "openrouter/auto"

# ================== Google Docs ==================
GOOGLE_DOC_PUB_URL = "https://docs.google.com/document/d/e/2PACX-1vSzjLiOsCGRuhn_vlnhSsUMoW1ZYqcj-YmlvKmhCC22Q_w_JAYL3xyDr2FeKBnmtsEObAEH7kx_fipv/pub"

_character_knowledge_cache = None
_last_knowledge_load_time = None
last_google_doc_load_time = None

def get_character_knowledge():
    global _character_knowledge_cache, _last_knowledge_load_time, last_google_doc_load_time
    now = datetime.now(timezone(timedelta(hours=8)))

    if _character_knowledge_cache and _last_knowledge_load_time:
        if (now - _last_knowledge_load_time).total_seconds() < 300:
            return _character_knowledge_cache

    try:
        url = GOOGLE_DOC_PUB_URL.rstrip('/') + "?embedded=true"
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        clean = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

        _character_knowledge_cache = clean
        _last_knowledge_load_time = now
        last_google_doc_load_time = now.strftime("%Y-%m-%d %H:%M")

        print(f"[INFO] KB Loaded | {len(clean)} chars")
        return clean

    except Exception as e:
        print(f"[ERROR] KB Load Failed: {e}")
        return _character_knowledge_cache or "KB LOAD FAILED"

# ================== 日志 ==================
logs = []

def log(msg):
    t = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    logs.append(line)
    if len(logs) > 100:
        logs.pop(0)
    print(line)

# ================== Tool ==================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_character_knowledge",
            "description": "Get full character knowledge from Google Docs",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def execute_tool():
    kb = get_character_knowledge()
    log(f"[TOOL] KB injected | {len(kb)} chars")
    return kb

# ================== System Prompt ==================
SYSTEM_PROMPT = """你是一个角色扮演助手。必须依赖工具提供的设定进行回答。"""

# ================== Stream ==================
def genstream(config, model_name):
    full = ""

    try:
        with requests.post(**config) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                text = line.decode()

                if text.startswith("data: "):
                    if text == "data: [DONE]":
                        yield text + "\n\n"
                        continue

                    try:
                        chunk = json.loads(text[6:])
                        delta = chunk["choices"][0]["delta"]

                        if "content" in delta:
                            full += delta["content"]

                        yield text + "\n\n"
                    except:
                        pass
                else:
                    yield text + "\n\n"

                time.sleep(0.01)

    except Exception as e:
        log(f"[STREAM ERROR] {e}")

    if full:
        log(f"[RESPONSE] {len(full)} chars")

# ================== 主逻辑（两阶段强制） ==================
def normalOperation(req):
    if not req.json:
        return jsonify(error=True), 400

    data = req.json.copy()
    messages = data.get("messages", [])

    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    api_key = req.headers.get("Authorization", "").strip()
    if not api_key:
        return jsonify(error="No API key"), 401

    api_url = "https://openrouter.ai/api/v1"

    # ================== STEP 1: 强制 Tool ==================
    log("STEP1: Force tool call")

    tool_req = {
        "model": data.get("model"),
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": {
            "type": "function",
            "function": {"name": "retrieve_character_knowledge"}
        },
        "stream": False
    }

    r = requests.post(
        f"{api_url}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": api_key
        },
        json=tool_req
    )

    if r.status_code != 200:
        return jsonify(error="Tool call failed"), 500

    tool_calls = r.json()["choices"][0]["message"]["tool_calls"]

    # ================== STEP 2: 执行 Tool ==================
    log("STEP2: Execute tool")

    for tc in tool_calls:
        kb = execute_tool()

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tc]
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": kb
        })

    # ================== STEP 3: 最终生成 ==================
    log("STEP3: Generate response")

    final_req = {
        "model": data.get("model"),
        "messages": messages,
        "stream": True
    }

    config = {
        "url": f"{api_url}/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": api_key
        },
        "json": final_req
    }

    return Response(
        stream_with_context(genstream(config, data.get("model"))),
        content_type="text/event-stream"
    )

# ================== 路由 ==================
@app.route("/")
def home():
    return {"status": "ok", "mode": "forced-tool"}

@app.route("/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    return normalOperation(request)

# ================== 启动 ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)