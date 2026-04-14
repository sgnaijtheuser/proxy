# ================================================
# OpenRouter Reverse Proxy - v3 Memory & Continuity
# ================================================

import json
import random
import string
import requests
import time
import hashlib
import os
import threading
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================== Configuration ==================
KEEP_RECENT_N  = int(os.environ.get("KEEP_RECENT_N",  "15"))   # messages kept in full
SUMMARY_BATCH  = int(os.environ.get("SUMMARY_BATCH",  "10"))   # min new msgs before summarizing
MAX_LOG_LINES  = 500
SGT            = timezone(timedelta(hours=8))

# ================== Upstash Redis ==================
UPSTASH_URL   = os.environ.get("UPSTASH_REDIS_REST_URL",   "")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

def _redis(command: list):
    """Execute a Redis command via Upstash REST API. Returns result or None on failure."""
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        log("[REDIS] ✗ Upstash not configured — UPSTASH_REDIS_REST_URL / TOKEN missing")
        return None
    try:
        r = requests.post(
            UPSTASH_URL,
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}", "Content-Type": "application/json"},
            json=command,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("result")
        log(f"[REDIS] ✗ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log(f"[REDIS] ✗ Exception: {e}")
    return None

# ================== Google Docs KB ==================
GOOGLE_DOC_PUB_URL = (
    "https://docs.google.com/document/d/e/"
    "2PACX-1vSzjLiOsCGRuhn_vlnhSsUMoW1ZYqcj-YmlvKmhCC22Q_w_JAYL3xyDr2FeKBnmtsEObAEH7kx_fipv/pub"
)
_kb_cache     = None
_kb_loaded_at = None

def get_character_knowledge() -> str:
    global _kb_cache, _kb_loaded_at
    now = datetime.now(SGT)
    if _kb_cache and _kb_loaded_at and (now - _kb_loaded_at).total_seconds() < 100:
        return _kb_cache
    try:
        r = requests.get(GOOGLE_DOC_PUB_URL + "?embedded=true", timeout=15)
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        text  = soup.get_text(separator="\n", strip=True)
        clean = "\n".join(l.strip() for l in text.splitlines() if l.strip())
        _kb_cache     = clean
        _kb_loaded_at = now
        log(f"[KB] ✓ Loaded {len(clean)} chars from Google Docs")
        return clean
    except Exception as e:
        log(f"[KB] ✗ Load failed: {e}")
        return _kb_cache or "KB LOAD FAILED"

# ================== Logging ==================
logs     = []
_log_lk  = threading.Lock()

def log(msg: str):
    t    = datetime.now(SGT).strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    with _log_lk:
        logs.append(line)
        if len(logs) > MAX_LOG_LINES:
            logs.pop(0)
    print(line, flush=True)

@app.route("/logs")
def view_logs():
    with _log_lk:
        lines = list(reversed(logs))
    body = "\n".join(lines)
    return (
        f'<!DOCTYPE html><html><head><title>Proxy Logs</title>'
        f'<meta http-equiv="refresh" content="5">'
        f'<style>'
        f'body{{background:#111;color:#ddd;font-family:monospace;font-size:12px;padding:16px;margin:0}}'
        f'pre{{white-space:pre-wrap;word-break:break-all}}'
        f'h3{{color:#888;margin-bottom:8px}}'
        f'</style></head><body>'
        f'<h3>Proxy Logs [{len(lines)} lines] — auto-refresh 5s</h3>'
        f'<pre>{body}</pre>'
        f'</body></html>'
    )

# ================== Session Management ==================
def load_session(sid: str) -> dict:
    result = _redis(["GET", f"session:{sid}"])
    if result:
        try:
            return json.loads(result)
        except Exception as e:
            log(f"[SESSION] ✗ Parse error {sid[:8]}: {e}")
    return {
        "session_id":      sid,
        "last_summary_at": 0,    # how many msgs have been summarized (index into stripped msg list)
        "rolling_summary": "",
        "current_state":   "",
        "turns":           0,
    }

def save_session(s: dict):
    sid = s["session_id"]
    s["saved_at"] = datetime.now(SGT).isoformat()
    result = _redis(["SET", f"session:{sid}", json.dumps(s, ensure_ascii=False)])
    if result == "OK":
        log(
            f"[SESSION] ✓ Saved {sid[:8]}… "
            f"turns={s.get('turns', 0)} "
            f"summarized_msgs={s.get('last_summary_at', 0)}"
        )
    else:
        log(f"[SESSION] ✗ Save failed {sid[:8]}: result={result}")

def derive_session_id(messages: list) -> str:
    """
    Session ID = hash(character_anchor + first_real_user_message).
    - character_anchor: first system or assistant message with real content (character card)
    - user_anchor: first user message that is NOT an XML/instruction wrapper
    This ensures same character + same conversation opener = same session,
    but a NEW conversation (different opener) gets a fresh session ID.
    """
    char_anchor = ""
    user_anchor = ""

    for m in messages:
        content = (m.get("content") or "").strip()
        role    = m.get("role", "")

        if not char_anchor and role in ("system", "assistant") and len(content) > 30:
            char_anchor = content[:200]

        # Skip XML/instruction wrappers injected by clients (e.g. Anime.gf)
        if not user_anchor and role == "user" and len(content) > 10 and not content.startswith("<"):
            user_anchor = content[:200]

        if char_anchor and user_anchor:
            break

    if not char_anchor and not user_anchor:
        return hashlib.sha256(json.dumps(messages[:3]).encode()).hexdigest()[:16]

    key = (char_anchor + "||" + user_anchor).encode()
    return hashlib.sha256(key).hexdigest()[:16]

@app.route("/sessions")
def view_sessions():
    keys = _redis(["KEYS", "session:*"]) or []
    rows = ""
    for key in sorted(keys):
        raw = _redis(["GET", key])
        if not raw:
            continue
        try:
            s   = json.loads(raw)
            sid = s.get("session_id", key.replace("session:", ""))
            rows += (
                f"<tr>"
                f"<td><a href='/session/{sid}' style='color:#6bf'>{sid[:8]}…</a></td>"
                f"<td>{s.get('turns', 0)}</td>"
                f"<td>{s.get('last_summary_at', 0)}</td>"
                f"<td>{'✓' if s.get('rolling_summary') else '–'}</td>"
                f"<td>{'✓' if s.get('current_state') else '–'}</td>"
                f"<td>{s.get('saved_at', '–')}</td>"
                f"<td><small>{(s.get('rolling_summary') or '')[:100]}</small></td>"
                f"<td><small>{(s.get('current_state') or '')[:100]}</small></td>"
                f"</tr>"
            )
        except Exception:
            pass
    return (
        f'<!DOCTYPE html><html><head><title>Sessions</title>'
        f'<meta http-equiv="refresh" content="10">'
        f'<style>'
        f'body{{background:#111;color:#ddd;font-family:monospace;font-size:12px;padding:16px}}'
        f'table{{border-collapse:collapse;width:100%}}'
        f'th,td{{border:1px solid #333;padding:5px 8px;text-align:left}}'
        f'th{{background:#222;color:#888}}'
        f'</style></head><body>'
        f'<h3 style="color:#888">Sessions ({len(keys)}) — auto-refresh 10s</h3>'
        f'<table>'
        f'<tr><th>ID</th><th>Turns</th><th>Summarized msgs</th>'
        f'<th>Summary</th><th>State</th><th>Saved</th>'
        f'<th>Summary preview</th><th>State preview</th></tr>'
        f'{rows}'
        f'</table></body></html>'
    )

@app.route("/session/<sid>")
def view_session(sid: str):
    s = load_session(sid)
    if not s.get("saved_at"):
        return f"Session {sid!r} not found in Redis", 404
    return (
        f'<!DOCTYPE html><html><head><title>Session {sid}</title>'
        f'<style>'
        f'body{{background:#111;color:#ddd;font-family:monospace;font-size:12px;padding:16px}}'
        f'pre{{white-space:pre-wrap;border:1px solid #333;padding:10px;background:#1a1a1a}}'
        f'</style></head><body>'
        f"<h3>Session {s.get('session_id','?')[:8]}…</h3>"
        f"<b>Turns:</b> {s.get('turns',0)} | "
        f"<b>Summarized up to msg:</b> {s.get('last_summary_at',0)} | "
        f"<b>Saved:</b> {s.get('saved_at','–')}"
        f"<h4>Rolling Summary</h4><pre>{s.get('rolling_summary','(none)')}</pre>"
        f"<h4>Current State</h4><pre>{s.get('current_state','(none)')}</pre>"
        f'</body></html>'
    )

# ================== Tool Definition ==================
TOOL_NAME = "retrieve_character_knowledge"
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Retrieve the full character knowledge base including personality, background, "
            "relationships, speech style, and world-building details. "
            "Always call this before generating a reply to ensure accurate character portrayal."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}]
TOOL_CHOICE_NAMED    = {"type": "function", "function": {"name": TOOL_NAME}}
TOOL_CHOICE_REQUIRED = "required"

def make_tcid() -> str:
    return "call_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=16))

def build_kb_content(session: dict) -> str:
    """Build the full context block injected as the tool result: KB + story summary + current state."""
    kb    = get_character_knowledge()
    parts = [f"=== CHARACTER KNOWLEDGE BASE ===\n{kb}"]
    if session.get("rolling_summary"):
        parts.append(f"=== STORY SO FAR ===\n{session['rolling_summary']}")
    if session.get("current_state"):
        parts.append(f"=== CURRENT STORY STATE ===\n{session['current_state']}")
    total = "\n\n".join(parts)
    log(
        f"[KB-CONTENT] Built {len(total)} chars — "
        f"kb={len(kb)} "
        f"summary={len(session.get('rolling_summary',''))} "
        f"state={len(session.get('current_state',''))}"
    )
    return total

# ================== Tool Call Attempt ==================
def try_tool_call(url, headers, messages, model, tool_choice):
    body = {"messages": messages, "tools": TOOLS, "tool_choice": tool_choice, "stream": False}
    if model:
        body["model"] = model
    tc_label = json.dumps(tool_choice) if isinstance(tool_choice, dict) else repr(tool_choice)
    log(f"[TOOL-REQ] → tool_choice={tc_label} | {len(messages)} msgs in context")
    try:
        r = requests.post(url, headers=headers, json=body, timeout=180)
        if r.status_code != 200:
            log(f"[TOOL-REQ] ✗ HTTP {r.status_code}: {r.text[:300]}")
            return None, None
        resp   = r.json()
        choice = resp["choices"][0]
        msg    = choice["message"]
        fin    = choice.get("finish_reason", "?")
        tcs    = msg.get("tool_calls") or []
        if tcs:
            names = [tc["function"]["name"] for tc in tcs]
            log(f"[TOOL-CALL] ✓ REAL TOOL CALL | finish_reason={fin} | tools={names}")
            return tcs, msg
        else:
            preview = repr((msg.get("content") or "")[:100])
            log(
                f"[TOOL-CALL] ✗ NO TOOL CALL RETURNED | finish_reason={fin} | "
                f"tool_choice={tc_label} | content_preview={preview}"
            )
            return None, None
    except Exception as e:
        log(f"[TOOL-REQ] ✗ Exception: {e}")
        return None, None

# ================== Streaming ==================
def genstream(url, headers, body, acc: dict):
    """Stream SSE chunks. Accumulates full text into acc['text'] when done."""
    full = ""
    try:
        with requests.post(url, headers=headers, json=body, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                text = raw.decode("utf-8")
                if text.startswith("data: "):
                    if text == "data: [DONE]":
                        yield text + "\n\n"
                        continue
                    try:
                        chunk = json.loads(text[6:])
                        delta = chunk["choices"][0]["delta"]
                        if delta.get("content"):
                            full += delta["content"]
                        yield text + "\n\n"
                    except Exception:
                        pass
                else:
                    yield text + "\n\n"
                time.sleep(0.005)
    except Exception as e:
        log(f"[STREAM] ✗ Error: {e}")
    acc["text"] = full
    if full:
        log(f"[STREAM] ✓ Complete — {len(full)} chars generated")
    else:
        log(f"[STREAM] ✗ Empty response from model")

# ================== Background: Summarization ==================
def _fmt(messages: list) -> str:
    """Format message list for LLM prompts, skipping tool messages."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        if role == "tool":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        out.append(f"[{role.upper()}]: {content[:400]}")
    return "\n".join(out)

def _llm_call(url, headers, model, prompt: str, max_tokens: int) -> str | None:
    """Make a simple non-streaming LLM call. Returns response text or None."""
    body = {
        "messages":   [{"role": "user", "content": prompt}],
        "stream":     False,
        "max_tokens": max_tokens,
    }
    if model:
        body["model"] = model
    try:
        r = requests.post(url, headers=headers, json=body, timeout=45)
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]                                                                                                             
            content = msg.get("content") or msg.get("reasoning_content") or ""                                                                                  
            content = content.strip()                                         
            if not content:                                                                                                                                     
                log(f"[LLM-AUX] ✗ Null/empty content | msg keys: {list(msg.keys())} | raw: {r.text[:300]}")
                return None                                                                                                                                     
            return content    
        log(f"[LLM-AUX] ✗ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log(f"[LLM-AUX] ✗ Exception: {e}")
    return None

def bg_summarize(sid: str, new_msgs: list, new_last_at: int, url, headers, model):
    """Background: extend rolling summary with newly archived messages."""
    log(f"[SUMMARY] Starting summarization for {sid[:8]} ({len(new_msgs)} new msgs)")
    s        = load_session(sid)
    existing = s.get("rolling_summary", "")
    formatted = _fmt(new_msgs)
    prompt = (
        (f"Previous summary:\n{existing}\n\n" if existing else "") +
        f"New messages to incorporate:\n{formatted}\n\n"
        "Write a concise 3-5 paragraph narrative summary of the full story so far. "
        "Cover: key events, decisions made, actions taken, where characters are, "
        "any unresolved threads. Be specific and factual. "
        "This will be injected as context to maintain roleplay continuity."
    )
    result = _llm_call(url, headers, model, prompt, 600)
    if result:
        fresh = load_session(sid)
        fresh["rolling_summary"] = result
        fresh["last_summary_at"] = new_last_at
        save_session(fresh)
        log(f"[SUMMARY] ✓ Done for {sid[:8]} — {len(result)} chars, now summarized up to msg {new_last_at}")
    else:
        log(f"[SUMMARY] ✗ Failed for {sid[:8]}")

# ================== Background: State Extraction ==================
def bg_extract_state(session: dict, response_text: str, recent_msgs: list, url, headers, model):
    """Background: extract the current story state from the latest exchange."""
    sid   = session["session_id"]
    convo = _fmt(recent_msgs[-6:])
    prompt = (
        f"Recent roleplay exchange:\n{convo}\n\n"
        f"Character's latest response:\n{response_text[:1200]}\n\n"
        "Based on the above, extract the current story state in under 120 words:\n"
        "- Where is the character right now (location / situation)?\n"
        "- What did the character just do or say?\n"
        "- What is the character's immediate intent or plan?\n"
        "- Any open commitments or story threads that must continue?\n"
        "Write in present tense. Be specific and factual."
    )
    result = _llm_call(url, headers, model, prompt, 250)
    if result:
        fresh          = load_session(sid)
        fresh["current_state"] = result
        fresh["turns"] = fresh.get("turns", 0) + 1
        save_session(fresh)
        log(f"[STATE] ✓ Updated for {sid[:8]}: {result[:80]}…")
    else:
        log(f"[STATE] ✗ Extraction failed for {sid[:8]}")

# ================== Main Handler ==================
def normalOperation(req):
    if not req.json:
        return jsonify(error=True), 400

    data     = req.json.copy()
    messages = list(data.get("messages", []))
    _INVALID = {"default", "Default", "auto", "openrouter/auto", "", None}
    model    = data.get("model")
    if model in _INVALID:
        model = None
        log(f"[REQUEST] model override → None (will use OpenRouter Default Model: DeepSeek V3.2)")

    # Derive session ID from raw messages (before stripping), using first substantial content
    sid     = derive_session_id(messages)
    session = load_session(sid)

    # Refresh trigger — clear state/summary if user message contains "refresh"
    last_user = next((m.get("content","") for m in reversed(messages)             
                    if m.get("role") == "user"), "")                            
    if "(refresh)" in (last_user or "").lower():                                  
      session["rolling_summary"] = ""                                           
      session["current_state"]   = ""                                           
      session["last_summary_at"] = 0                                            
      save_session(session)                                                     
      log(f"[REFRESH] ✓ Session {sid[:8]} state cleared by trigger")  

    # Strip all system messages — the proxy owns context injection
    VALID_ROLES = {"user", "assistant", "tool"}
    messages    = [m for m in messages if m.get("role") in VALID_ROLES]

    # Inject KB rules as system message at position 0 (highest priority, before conversation history).    
    # This ensures new Google Doc rules take effect immediately, even when conversation history contains old behavioral patterns that would otherwise override the KB tool result.  
    # kb_system = get_character_knowledge()
    # messages = [{"role": "system", "content": f"=== CHARACTER RULES (AUTHORITATIVE) ===\n{kb_system}"}] + messages                                  
    # log(f"[SYSTEM] Injected {len(kb_system)} chars KB as system message at position 0")           

    api_key = req.headers.get("Authorization", "").strip()
    if not api_key:
        return jsonify(error="No API key"), 401

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    hdrs    = {"Content-Type": "application/json", "Authorization": api_key}

    log("=" * 60)
    log(
        f"[REQUEST] session={sid[:8]} | model={model or '(default)'} | "
        f"raw_msgs={len(messages)} | turns={session.get('turns', 0)}"
    )

    # ===== Summarization Check =====
    n        = len(messages)
    old_msgs = messages[: n - KEEP_RECENT_N] if n > KEEP_RECENT_N else []

    if old_msgs:
        last_at       = session.get("last_summary_at", 0)
        to_summarize  = old_msgs[last_at:]
        new_last_at   = len(old_msgs)
        if len(to_summarize) >= SUMMARY_BATCH:
            log(
                f"[SUMMARY] Queuing background summarization — "
                f"msgs {last_at}→{new_last_at} ({len(to_summarize)} new msgs)"
            )
            threading.Thread(
                target=bg_summarize,
                args=(sid, to_summarize, new_last_at, api_url, hdrs, model),
                daemon=True,
            ).start()
        else:
            log(
                f"[SUMMARY] Holding off — only {len(to_summarize)} new archivable msgs "
                f"(need ≥{SUMMARY_BATCH})"
            )
    else:
        log(f"[SUMMARY] Not needed — {n} msgs total, all within recent window ({KEEP_RECENT_N})")

    # Trim to recent window only
    msgs = messages[-KEEP_RECENT_N:] if n > KEEP_RECENT_N else messages
    log(
        f"[CONTEXT] Sending {len(msgs)} msgs to LLM | "
        f"has_summary={bool(session.get('rolling_summary'))} | "
        f"has_state={bool(session.get('current_state'))}"
    )

    # Log last few turns for debug
    log("[MSGS] Last 3 messages:")
    for m in msgs[-3:]:
        role    = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = str(content)[:200]
        log(f"  [{role}] {repr(content[:150])}")

    # ===== Step 1: Force tool call (named function) =====
    log("[STEP1] Attempting tool_choice=named function…")
    tcs, asst_msg = try_tool_call(api_url, hdrs, msgs, model, TOOL_CHOICE_NAMED)

    # ===== Step 1b: Retry with "required" =====
    if not tcs:
        log("[STEP1b] Retrying with tool_choice='required'…")
        tcs, asst_msg = try_tool_call(api_url, hdrs, msgs, model, TOOL_CHOICE_REQUIRED)

    # ===== Step 2: Execute tool or inject directly =====
    kb_content = build_kb_content(session)   # KB + summary + state

    if tcs:
        # Model did a real tool call — append its assistant message + real tool result
        log(f"[STEP2] ✓ REAL TOOL CALL — executing {len(tcs)} tool(s)")
        msgs.append({
            "role":       "assistant",
            "content":    asst_msg.get("content"),
            "tool_calls": tcs,
        })
        for tc in tcs:
            log(f"[STEP2] Executing tool '{tc['function']['name']}' → injecting {len(kb_content)} chars")
            msgs.append({
                "role":        "tool",
                "tool_call_id": tc["id"],
                "content":     kb_content,
            })
    else:
        # Both tool_choice strategies failed — fabricate tool call to inject context
        log("[STEP2] ✗ BOTH STRATEGIES FAILED — falling back to fake tool message injection")
        fake_id = make_tcid()
        msgs.append({
            "role":    "assistant",
            "content": None,
            "tool_calls": [{
                "id":       fake_id,
                "type":     "function",
                "function": {"name": TOOL_NAME, "arguments": "{}"},
            }],
        })
        msgs.append({
            "role":        "tool",
            "tool_call_id": fake_id,
            "content":     kb_content,
        })
        log(f"[STEP2] Injected {len(kb_content)} chars via fake tool message")

    # ===== Step 3: Stream final response =====
    log("[STEP3] Streaming final response…")
    final_body = {"messages": msgs, "stream": True, "max_tokens": 400}
    if model:
        final_body["model"] = model

    acc           = {"text": ""}
    snapshot_msgs = list(msgs)    # snapshot before the generator mutates anything

    def full_gen():
        yield from genstream(api_url, hdrs, final_body, acc)
        # After stream finishes, kick off background state extraction
        if acc["text"]:
            threading.Thread(
                target=bg_extract_state,
                args=(session, acc["text"], snapshot_msgs, api_url, hdrs, model),
                daemon=True,
            ).start()
            log("[STEP3] ✓ State extraction queued in background")
        else:
            log("[STEP3] ✗ Empty stream — state extraction skipped")

    return Response(stream_with_context(full_gen()), content_type="text/event-stream")

# ================== Routes ==================
@app.route("/")
def home():
    return jsonify({
        "status":  "ok",
        "version": "v3-memory-redis",
        "storage": "Upstash Redis" if UPSTASH_URL else "⚠ Upstash not configured",
        "config": {
            "keep_recent_n": KEEP_RECENT_N,
            "summary_batch": SUMMARY_BATCH,
        },
        "debug_endpoints": {
            "/logs":         "Live log viewer (auto-refresh 5s)",
            "/sessions":     "All sessions overview (auto-refresh 10s)",
            "/session/<id>": "Full session detail",
        },
    })
@app.route("/reset-session/<sid>", methods=["POST", "GET"])
def reset_session(sid: str):                                                  
    s = load_session(sid)
    if not s.get("saved_at"):                                                 
         return f"Session {sid!r} not found", 404
    s["rolling_summary"] = ""                                                 
    s["current_state"]   = ""                                                 
    s["last_summary_at"] = 0                                                  
    save_session(s)                                                           
    return jsonify(ok=True, session=sid, msg="State and summary cleared") 
@app.route("/chat/completions",    methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    return normalOperation(request)

# ================== Startup ==================
if __name__ == "__main__":
    log(f"[STARTUP] v3-memory-redis | keep_recent={KEEP_RECENT_N} | summary_batch={SUMMARY_BATCH}")
    log(f"[STARTUP] Upstash URL={'SET' if UPSTASH_URL else '✗ MISSING'} | Token={'SET' if UPSTASH_TOKEN else '✗ MISSING'}")
    app.run(host="0.0.0.0", port=5000)
