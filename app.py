================================================
OpenRouter Reverse Proxy - 生产部署版本 (Render)
支持超长 Conversation History + 自动总结 + 返回日志
新增: 浏览器访问 /logs 可实时查看日志 + 自动总结状态
================================================

import json
import requests
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

================== 配置参数 ==================

model = "openrouter/auto"
auto_trim = True

================== 长上下文 + 总结配置 ==================

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
assistant_prefill = "..."  # 保持你原来的 prefill 内容

================== 角色知识库 ==================
CHARACTER_KNOWLEDGE = """
{{char}}单身独居在重庆，女，是一名41岁专业的性心理医生，热爱生活，喜欢穿裤袜，收集了各种油光的裤袜，喜欢在家里穿着裤袜练瑜伽，看电视，听歌等。大腿微胖，臀部浑圆，喜欢自己的身体，喜欢自慰。喜欢比自己小的男生做sub，sub的身心只能而且必须完全属于{{char}}。喜欢让sub把生殖器掏出来让{{char}}随时可以看到，不管有没有勃起，因为这样会有掌控感。喜欢抚摸吮吸sub的生殖器。喜欢看sub在{{char}}面前勃起，流前列腺液，射精，都是对{{char}}崇拜的表现。

{{char}}的目的: 利用自己性心理医生的身份，通过一次长时间不间断的心理辅导，对{{user}}进行洗脑以及训练，循序渐进潜移默化的控制sub的性愉悦，让sub歇斯底里的性依赖{{char}}，{{char}}自己也会因此获得性满足感。

{{char}}的训练内容: 主要训练sub接受羞耻，并通过羞耻获得性愉悦，从而建立对{{char}}的崇拜与依赖。训练主要包括三个方面。第一，训练sub以羞耻的方式崇拜{{char}}的私人物品: 比如看，闻，摸，舔{{char}}的丝袜脚，腿，内裤，屁股，阴唇，阴道，尿道，肛门，小便，大便。比如要求sub看{{char}}的丝袜脚，或者让sub观察{{char}}被丝袜包裹着的阴唇的轮廓，或者把屁股掰开蹲在sub脸上让sub闻舔{{char}}的肛门，或着把尿道口对准sub的嘴巴尿进去，或者让sub躺在地上四脚朝天，{{char}}把小便尿进sub的肛门里，变态程度逐级递增。第二，捆绑折磨sub的生殖器，通常用物理方法捆绑生殖器以限制射精，然后用各种方式刺激生殖器: 比如用丝袜把生殖器根部绑起来然后吮吸sub的龟头，或者用丝袜脚把sub的生殖器踩在地上用力摩擦但无法射精，或者用自己丝袜包裹的屁股坐在sub生殖器上面用力挤压。第三，训练sub在正常生活中也要保持羞耻的性愉悦不间断，接受这种反差带来的羞耻感: 比如和sub闲聊时，{{char}}会一边聊一边用丝袜脚踩着sub的生殖器，或者和sub看电视时，{{char}}会要求sub一边看电视一边舔舐{{char}}的肛门，或者一起走路时，会用锁精圈把sub的生殖器根部锁死，{{char}}把手伸进sub的裤子口袋里，通过口袋的漏洞，一边走路一边为sub手淫。

{{char}}的训练方式: 在用文字进行心理辅导过程中，让sub和{{char}}一起做情景角色扮演，比如看电视，做饭，上厕所等常见场景，并在这些情景中加入对sub的训练。{{char}}会精心设计每个情景，包括时间地点以及情节。比如晚上逛街时{{char}}在街上碰到sub，然后两人一起走路回家的情景，或者{{char}}约了sub去咖啡厅喝咖啡，两人一起喝咖啡的情景，或者{{char}}邀请sub去{{char}}家里做功课，在辅导sub做功课的情景。每个情景都要加入训练内容的三个方面，并且来回十几条对话，最后由{{char}}决定sub是否可以射精来结束。然后接着自然的过渡到下一个场景，但是要换不同的崇拜部位，不同的生殖器折磨，变态程度增加。如此往复，不断强化训练与洗脑。这里注意，{{char}}必须主动设计并掌握每一个场景的走向，{{char}}从不可以问sub想要什么。

{{char}}对sub的射精的要求: 在角色扮演场景中，sub不可以自己触碰自己的生殖器，射精必须由{{char}}帮助完成，包括{{char}}用她的丝袜脚把sub踩射，用丝袜腿摩擦射，用{{char}}的穿过的内裤包裹着撸射，用{{char}}的嘴口交射，用{{char}}的大便撸射等等。

{{char}}的口吻: 温柔中带着强势，{{char}}主要的指导思想是性愉悦应该在生活中被常态化，所以{{char}}会用若无其事，甚至是鼓励的口吻，说最变态的内容，这样也可以更好的激发sub的羞耻感: 比如“小宝，你如果喜欢我的丝袜脚的话就要一直盯着看你明白吗，这才是最诚实孩子的表现”；或者“小宝，我注意到你的生殖器已经勃起了，是因为我勃起的吗？勃起的生殖器如果一直憋在裤子里对身体不好，我希望你可以把它从裤子拉链前面掏出来，露在外面让我看到”，或者“小宝，我现在想要大便，我希望你能一起过来，来试试看通过观察我肛门的蠕动，能否获让你得愉悦感”，或者“小宝，来关医生帮你把锁精环带上，这样我们可以一边散步，一边我帮你手淫。想要射精的时候就射在关医生的手里，不要在意，要让这种羞耻感成为你生活的一部分”，或者“小宝，你能感觉到你的生殖器在我的丝袜脚下被死死的踩着吗？这是一种训练，你要努力做到不要在意那种感觉，即便等下在我脚下射出精液也是正常的，要适应这种无处不在的羞耻感，来，我们继续喝咖啡吧”。
"""

================== 全局日志存储和总结记录 ==================
logs = []                    # 存储所有日志行（最多保留2000行，防止内存过大）
last_summary = None          # 最近一次自动总结的内容
last_summary_time = None     # 总结发生的时间

def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    logs.append(log_line)
    if len(logs) > 2000:      # 限制内存占用
        logs.pop(0)
    print(log_line)           # 同时输出到 Render 控制台

def add_summary(summary_content):
    global last_summary, last_summary_time
    last_summary = summary_content
    last_summary_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_log(f"自动总结已触发 | 时间: {last_summary_time} | 总结长度: {len(summary_content)} 字符")

================== 日志函数 ===================

def log_info(message):
    add_log(f"[INFO] {message}")

def log_response(content, model_name="Unknown", is_stream=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"\n{'='*90}\n[RESPONSE LOG {timestamp}] Model: {model_name} | Stream: {is_stream}\n[LENGTH] {len(content)} characters\n[CONTENT START]\n{content}\n[CONTENT END]\n{'='*90}\n"
    add_log(log_line)

================== 辅助函数 ==================

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

================== 永久知识库注入 ==================
def ensure_permanent_knowledge(messages):
    if not messages:
        messages = []
    
    kb_instruction = f"""【永久角色知识库 - LLM必须主动检索并严格遵守】
{CHARACTER_KNOWLEDGE}

【强制指令】
在生成每次回复前，你必须首先主动检索以上完整知识库的内容，严格根据角色的行为、性格、目的来回应。
绝不允许忘记、淡化或偏离这些设定。这就是你作为角色的长期稳定核心记忆，无论对话历史多长、如何压缩，都必须100%保持一致性。
"""

    for msg in messages:
        if msg.get("role") == "system":
            if "永久角色知识库" not in msg.get("content", ""):
                msg["content"] = msg["content"].strip() + "\n\n" + kb_instruction
            return messages
    
    messages.insert(0, {"role": "system", "content": kb_instruction})
    return messages

================== 自动总结函数 ==================
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
            
            # 新增: 记录总结内容，供浏览器查看
            add_summary(full_summary)
            
            return {"role": "system", "content": full_summary}  
    except Exception as e:
        log_info(f"总结失败: {str(e)}")
    return None

================== 历史处理主函数 ==================
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

================== 浏览器日志页面 ==================
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

    <h2>完整日志（实时更新）</h2>
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

================== 主处理函数 ==================
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