#!/usr/bin/env python3
"""OVMS OpenAI 协议对拍测试套件 (AIBOX 客户接口交付验证).

逐字段断言 OVMS 的 OpenAI 兼容行为, 把与 OpenAI 协议的差异变成红/绿.
覆盖客户 4 接口需求 + fork 的兼容性改造 (错误对象/状态码/id 注入/
response_format 校验/n 语义).

用法:
    MODEL=Qwen3.5-4B python3 test_openai_protocol.py
    BASE_URL=http://192.168.0.78:8000 API_KEY=xxx MODEL=Qwen3-VL-8B python3 test_openai_protocol.py

环境变量:
    BASE_URL   OVMS 服务地址 (默认 http://localhost:8000)
    API_KEY    API key (服务端配置了鉴权时必填)
    MODEL      已加载的模型名 (必填)
"""

import json
import os
import re
import sys

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "")

PASSED = []
FAILED = []


def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def chat(body=None, path="/v1/chat/completions", raw=None, timeout=120):
    """POST 一个 chat completions 请求; raw 用于构造非法 JSON."""
    headers = {"Content-Type": "application/json", **auth_headers()}
    if raw is not None:
        return requests.post(BASE + path, data=raw, headers=headers, timeout=timeout)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    payload.update(body or {})
    return requests.post(BASE + path, json=payload, headers=headers, timeout=timeout)


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


# ---------- 接口 1: 模型推理请求 ----------

def test_a1_response_id():
    """A1: 非流式响应必须有 chatcmpl- 前缀的 id."""
    r = chat({})
    d = r.json()
    cid = d.get("id", "")
    check("A1 响应含 id 且形如 chatcmpl-", isinstance(cid, str) and cid.startswith("chatcmpl-"), f"id={cid!r}")


def test_a2_streaming_same_id():
    """A2: 流式各 chunk 复用同一个 id."""
    r = chat({"stream": True}, timeout=120)
    ids = set()
    for line in r.text.splitlines():
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
            try:
                ids.add(json.loads(line[5:]).get("id"))
            except json.JSONDecodeError:
                pass
    check("A2 流式 chunk 共用同一 id", len(ids) == 1 and None not in ids, f"ids={ids}")


def test_a3_error_object():
    """A3: 404 错误体必须是 OpenAI 错误对象 (4 字段)."""
    r = requests.post(BASE + "/v1/chat/completions",
                      json={"model": "no-such-model-xyz", "messages": [{"role": "user", "content": "x"}]},
                      headers=auth_headers(), timeout=30)
    check("A3 404 状态码", r.status_code == 404, f"实际 {r.status_code}")
    err = r.json().get("error")
    ok = isinstance(err, dict) and {"message", "type", "param", "code"} <= set(err.keys())
    check("A3 错误体为 OpenAI 对象", ok, f"error={err!r}")


def test_a4_message_fields():
    """A4: message 含 refusal 字段, 无工具调用时 tool_calls 为 null."""
    d = chat({"max_tokens": 8}).json()
    msg = d["choices"][0]["message"]
    check("A4 message 含 refusal", "refusal" in msg)
    check("A4 无工具调用时 tool_calls 为 null", msg.get("tool_calls") is None, f"tool_calls={msg.get('tool_calls')!r}")


def test_a5_usage_details():
    """A5: usage 含 prompt_tokens_details."""
    u = chat({}).json().get("usage", {})
    check("A5 usage 含 prompt_tokens_details", "prompt_tokens_details" in u, f"usage={u!r}")


def test_a6_v1_v3_parity():
    """A6: /v1 与 /v3 前缀行为一致 (fork 保留双前缀)."""
    r1 = chat({}, path="/v1/chat/completions")
    r3 = chat({}, path="/v3/chat/completions")
    check("A6 /v1 与 /v3 均 200 且都有 id",
          r1.status_code == 200 and r3.status_code == 200
          and r1.json().get("id", "").startswith("chatcmpl-")
          and r3.json().get("id", "").startswith("chatcmpl-"),
          f"v1={r1.status_code} v3={r3.status_code}")


def test_a7_n_multiple_candidates():
    """A7 (fork 4.9): n>1 返回 n 个候选且不报错 (OpenAI 语义)."""
    r = chat({"n": 2, "max_tokens": 8})
    if r.status_code != 200:
        check("A7 n=2 返回 200", False, f"实际 {r.status_code}: {r.text[:200]}")
        return
    choices = r.json().get("choices", [])
    check("A7 n=2 返回 2 个候选", len(choices) == 2, f"实际 {len(choices)} 个")


# ---------- 接口 2: 可用模型查询 ----------

def test_b1_models_list():
    """B1: GET /v1/models 返回标准 list 形状且含当前模型."""
    r = requests.get(BASE + "/v1/models", headers=auth_headers(), timeout=30)
    ok = r.status_code == 200
    data = r.json().get("data", []) if ok else []
    has_model = any(m.get("id") == MODEL for m in data)
    check("B1 /v1/models 200 且包含当前模型", ok and has_model,
          f"status={r.status_code} ids={[m.get('id') for m in data][:5]}")


# ---------- 接口 3: 健康检查 ----------

def test_c1_health():
    """C1: KServe v2 健康端点全部 200."""
    for path in ["/v2/health/live", "/v2/health/ready", f"/v2/models/{MODEL}/ready"]:
        r = requests.get(BASE + path, timeout=30)
        check(f"C1 {path} 200", r.status_code == 200, f"实际 {r.status_code}")


# ---------- 错误与状态码 (fork 4.1/4.2) ----------

def test_d1_invalid_json_400():
    """D1: 非法 JSON 返回 400 (旧版 412)."""
    r = chat(raw="{bad json")
    check("D1 非法 JSON 返回 400", r.status_code == 400, f"实际 {r.status_code}")


def test_d2_unauthorized_401():
    """D2: 鉴权失败返回 401 + invalid_api_key (需服务端配置 api_key)."""
    r = requests.post(BASE + "/v1/chat/completions",
                      json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
                      headers={"Authorization": "Bearer deliberately-wrong-key"}, timeout=30)
    if r.status_code == 401:
        err = r.json().get("error", {})
        check("D2 401 + code=invalid_api_key", err.get("code") == "invalid_api_key", f"error={err!r}")
    else:
        check("D2 401 + code=invalid_api_key", False,
              f"实际 {r.status_code} (服务端未配置 api_key 时跳过此项)")


def test_d3_missing_messages_400():
    """D3: 缺 messages 返回 400 + invalid_parameter."""
    r = requests.post(BASE + "/v1/chat/completions",
                      json={"model": MODEL}, headers=auth_headers(), timeout=30)
    ok = r.status_code == 400
    check("D3 缺 messages 返回 400", ok, f"实际 {r.status_code}")


# ---------- fork 4.5: response_format ----------

def test_e1_response_format_bad_type():
    """E1 (fork 4.5): response_format 非法 type 返回 400 (旧版静默 200)."""
    r = chat({"response_format": {"type": "banana"}, "max_tokens": 8})
    check("E1 非法 response_format.type 返回 400", r.status_code == 400,
          f"实际 {r.status_code}: {r.text[:200]}")


def test_e2_response_format_json_object():
    """E2 (fork 4.5): json_object 真实强制输出为纯 JSON (旧版静默忽略).

    提示词要求模型先说一句自然语言再给 JSON: 无强制时模型会输出
    "Sure! {...}", 解析失败; 有强制时 XGrammar 只允许 JSON token.
    """
    r = chat({"response_format": {"type": "json_object"},
              "messages": [{"role": "user",
                            "content": "First greet me in one short sentence, then return a JSON object with key ok set to true"}],
              "max_tokens": 96})
    if r.status_code != 200:
        check("E2 json_object 200", False, f"实际 {r.status_code}: {r.text[:200]}")
        return
    content = r.json()["choices"][0]["message"]["content"]
    try:
        json.loads(content)
        check("E2 json_object 产出可解析 JSON", True)
    except json.JSONDecodeError as e:
        check("E2 json_object 产出可解析 JSON", False, f"content={content[:120]!r} ({e})")


def test_e3_response_format_json_schema():
    """E3: json_schema 结构化输出按 schema 产出."""
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    r = chat({"response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}},
              "messages": [{"role": "user", "content": "Return ok=true as JSON"}],
              "max_tokens": 64})
    if r.status_code != 200:
        check("E3 json_schema 200", False, f"实际 {r.status_code}: {r.text[:200]}")
        return
    content = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        check("E3 json_schema 产出符合 schema 的 JSON", parsed.get("ok") is True, f"parsed={parsed!r}")
    except json.JSONDecodeError as e:
        check("E3 json_schema 产出符合 schema 的 JSON", False, f"content={content[:120]!r} ({e})")


# ---------- 接口 4: 请求详情查询 (推理/排队) ----------

def _status_fields(snap):
    """校验状态对象字段完整且类型正确; 返回 (错误列表, 有效)."""
    required = ["id", "model", "status", "queue_position", "in_flight",
                "submitted_at_us", "scheduled_at_us", "started_at_us",
                "completed_at_us", "wait_ms", "ttft_ms", "error"]
    missing = [k for k in required if k not in snap]
    if missing:
        return f"缺字段: {missing}", False
    if snap["status"] not in ("queued", "running", "completed", "failed"):
        return f"非法 status: {snap['status']!r}", False
    return None, True


def test_f1_unary_completed_status():
    """F1: 非流式请求办结后查状态: completed + 精确时间戳 + ttft_ms/wait_ms."""
    r = chat({})
    if r.status_code != 200:
        check("F1 非流式 200", False, f"实际 {r.status_code}: {r.text[:200]}")
        return
    rid = r.json().get("id", "")
    check("F1 响应含 id", bool(rid.startswith("chatcmpl-")), f"id={rid!r}")
    sr = requests.get(f"{BASE}/v1/requests/{rid}", headers=auth_headers(), timeout=15)
    if sr.status_code != 200:
        check("F1 状态查询 200", False, f"实际 {sr.status_code}: {sr.text[:200]}")
        return
    snap = sr.json()
    err, ok = _status_fields(snap)
    check("F1 状态字段完整且类型合法", ok, err or f"snap={snap!r}")
    check("F1 id 一致", snap.get("id") == rid, f"请求 id={rid} 状态 id={snap.get('id')}")
    check("F1 status=completed", snap.get("status") == "completed", f"status={snap.get('status')}")
    check("F1 queue_position 终态为 null", snap.get("queue_position") is None,
          f"queue_position={snap.get('queue_position')!r}")
    sub, sch, sta, com = (snap.get(k) or 0 for k in
                          ("submitted_at_us", "scheduled_at_us", "started_at_us", "completed_at_us"))
    check("F1 时间戳单调 (sub<=sch<=sta<=com)", sub <= sch <= sta <= com,
          f"{sub},{sch},{sta},{com}")
    check("F1 ttft_ms 有精确值", isinstance(snap.get("ttft_ms"), int) and snap["ttft_ms"] >= 0,
          f"ttft_ms={snap.get('ttft_ms')!r}")
    check("F1 wait_ms>=0", isinstance(snap.get("wait_ms"), int) and snap["wait_ms"] >= 0,
          f"wait_ms={snap.get('wait_ms')!r}")


def test_f2_unknown_id_404():
    """F2: 未知 id → 404 + OpenAI 错误对象."""
    r = requests.get(f"{BASE}/v1/requests/chatcmpl-does-not-exist-000000",
                     headers=auth_headers(), timeout=15)
    check("F2 未知 id 返回 404", r.status_code == 404, f"实际 {r.status_code}")
    try:
        err = r.json().get("error", {})
        check("F2 错误体为 OpenAI 对象", isinstance(err, dict) and "message" in err and "code" in err,
              f"error={err!r}")
    except Exception as e:
        check("F2 错误体为 OpenAI 对象", False, f"解析失败: {e}")


def test_f3_in_flight_list():
    """F3: 在办列表 GET /v1/requests 结构合法."""
    r = requests.get(f"{BASE}/v1/requests", headers=auth_headers(), timeout=15)
    if r.status_code != 200:
        check("F3 列表 200", False, f"实际 {r.status_code}: {r.text[:200]}")
        return
    body = r.json()
    check("F3 含 in_flight 字段", isinstance(body.get("in_flight"), int),
          f"body={body!r}")
    check("F3 含 requests 数组", isinstance(body.get("requests"), list),
          f"body={body!r}")


def test_f4_streaming_midflight():
    """F4: 流式请求在办中查状态: running/queued + started 已观测; 办结后 completed+ttft."""
    payload = {"model": MODEL, "stream": True,
               "messages": [{"role": "user", "content": "Count from 1 to 30"}],
               "max_tokens": 400}
    headers = {"Content-Type": "application/json", **auth_headers()}
    rid = None
    mid = None
    with requests.post(BASE + "/v1/chat/completions", json=payload, headers=headers,
                       stream=True, timeout=120) as s:
        if s.status_code != 200:
            check("F4 流式 200", False, f"实际 {s.status_code}: {s.text[:200]}")
            return
        for line in s.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            rid = chunk.get("id")
            # 收到首个内容 chunk 后立即查状态 (生成仍在进行)
            if mid is None:
                sr = requests.get(f"{BASE}/v1/requests/{rid}", headers=auth_headers(), timeout=15)
                if sr.status_code == 200:
                    mid = sr.json()
            # 注意: 必须继续读到底 [DONE] 再退出, 提前 break 会断开 SSE 连接,
            # 服务端把该请求视为客户端取消 (failed/CANCELLED), 收不到 completed.
    if not rid:
        check("F4 流式返回 id", False, "未收到任何 chunk")
        return
    if mid is None:
        check("F4 在办查询 200", False, "查询返回非 200")
        return
    err, ok = _status_fields(mid)
    check("F4 在办状态字段完整", ok, err or f"snap={mid!r}")
    check("F4 在办 status 为 queued/running", mid.get("status") in ("queued", "running"),
          f"status={mid.get('status')}")
    check("F4 在办 started_at_us 已观测", isinstance(mid.get("started_at_us"), int),
          f"started_at_us={mid.get('started_at_us')!r}")
    check("F4 在办 completed_at_us 为 null", mid.get("completed_at_us") is None,
          f"completed_at_us={mid.get('completed_at_us')!r}")
    # 流结束 (with 块退出即读完) 后再查一次: 终态 + ttft
    import time
    time.sleep(0.5)
    fr = requests.get(f"{BASE}/v1/requests/{rid}", headers=auth_headers(), timeout=15)
    if fr.status_code != 200:
        check("F4 终态查询 200", False, f"实际 {fr.status_code}")
        return
    final = fr.json()
    check("F4 终态 status=completed", final.get("status") == "completed",
          f"status={final.get('status')}")
    check("F4 终态 ttft_ms 有精确值",
          isinstance(final.get("ttft_ms"), int) and final["ttft_ms"] >= 0,
          f"ttft_ms={final.get('ttft_ms')!r}")


def test_f5_post_not_allowed():
    """F5: POST /v1/requests → 400 + OpenAI 错误对象 (仅 GET 可用, 与 /v1/models 一致)."""
    r = requests.post(BASE + "/v1/requests", headers=auth_headers(),
                      json={"id": "x"}, timeout=15)
    check("F5 POST /v1/requests 返回 400", r.status_code == 400, f"实际 {r.status_code}")
    try:
        err = r.json().get("error", {})
        check("F5 错误体为 OpenAI 对象", isinstance(err, dict) and "message" in err,
              f"error={err!r}")
    except Exception as e:
        check("F5 错误体为 OpenAI 对象", False, f"解析失败: {e}")


def main():
    if not MODEL:
        print("错误: 必须通过环境变量 MODEL 指定已加载的模型名")
        sys.exit(2)
    print(f"对拍目标: {BASE}  模型: {MODEL}  鉴权: {'开' if API_KEY else '关'}")
    print()

    tests = [
        ("接口1 推理请求", [test_a1_response_id, test_a2_streaming_same_id, test_a3_error_object,
                           test_a4_message_fields, test_a5_usage_details, test_a6_v1_v3_parity,
                           test_a7_n_multiple_candidates]),
        ("接口2 模型查询", [test_b1_models_list]),
        ("接口3 健康检查", [test_c1_health]),
        ("接口4 请求详情查询", [test_f1_unary_completed_status, test_f2_unknown_id_404,
                               test_f3_in_flight_list, test_f4_streaming_midflight,
                               test_f5_post_not_allowed]),
        ("错误与状态码", [test_d1_invalid_json_400, test_d2_unauthorized_401, test_d3_missing_messages_400]),
        ("response_format", [test_e1_response_format_bad_type, test_e2_response_format_json_object,
                             test_e3_response_format_json_schema]),
    ]
    for group, fns in tests:
        print(f"[{group}]")
        for fn in fns:
            try:
                fn()
            except Exception as e:
                check(fn.__name__, False, f"异常: {e}")
        print()

    total = len(PASSED) + len(FAILED)
    print(f"===== 结果: {len(PASSED)}/{total} 通过 =====")
    if FAILED:
        print("失败项:")
        for name, detail in FAILED:
            print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
