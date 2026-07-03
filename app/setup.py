"""Interactive setup: collect + persist the gateway key and subscription token.

Two secrets only: the gateway API key (clients send it) and the subscription token
(from `claude setup-token`). The page is always editable — no change-lock.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from . import config
from .log import configure as log_reconfigure

router = APIRouter()


def _persist(values: dict[str, str]) -> None:
    for k, v in values.items():
        if v:
            os.environ[k] = v
    if config.CONFIG_FILE:
        path = Path(config.CONFIG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    existing[key.strip()] = val
        existing.update({k: v for k, v in values.items() if v})
        path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")
        os.chmod(path, 0o600)
    config.reload()


_STYLE = """
:root{
 --primary:#6750A4;--on-primary:#fff;--primary-container:#EADDFF;--on-primary-container:#21005D;
 --surface:#FEF7FF;--surface-container:#F3EDF7;--surface-variant:#E7E0EC;--on-surface:#1D1B20;
 --on-surface-variant:#49454F;--outline:#79747E;--error:#B3261E;
}
@media(prefers-color-scheme:dark){:root{
 --primary:#D0BCFF;--on-primary:#381E72;--primary-container:#4F378B;--on-primary-container:#EADDFF;
 --surface:#141218;--surface-container:#211F26;--surface-variant:#49454F;--on-surface:#E6E0E9;
 --on-surface-variant:#CAC4D0;--outline:#938F99;--error:#F2B8B5;
}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:400 15px/1.5 Roboto,system-ui,sans-serif;
 background:var(--surface);color:var(--on-surface);display:flex;justify-content:center;padding:32px 16px}
.wrap{width:100%;max-width:560px}
.bar{display:flex;align-items:center;gap:14px;margin-bottom:24px;padding:4px}
.bar .logo{width:44px;height:44px;border-radius:14px;background:var(--primary);color:var(--on-primary);
 display:grid;place-items:center;font-size:22px}
.bar h1{font-size:22px;font-weight:500;margin:0;letter-spacing:.1px}
.card{background:var(--surface-container);border-radius:28px;padding:28px 24px;
 box-shadow:0 1px 3px rgba(0,0,0,.15),0 4px 12px rgba(0,0,0,.08)}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;
 padding:6px 14px;border-radius:100px;background:var(--primary-container);color:var(--on-primary-container);margin-bottom:20px}
.field{position:relative;margin:26px 0}
.field input,.field select{width:100%;padding:16px;font:inherit;color:var(--on-surface);background:var(--surface-container);
 border:1px solid var(--outline);border-radius:12px;outline:none}
.field input:focus,.field select:focus{border:2px solid var(--primary);padding:15px}
.field label{position:absolute;top:-9px;left:12px;padding:0 6px;font-size:12px;
 background:var(--surface-container);color:var(--on-surface-variant)}
.field input:focus+label{color:var(--primary)}
.hint{font-size:12.5px;color:var(--on-surface-variant);margin:6px 4px 0}
.code{background:var(--surface-variant);color:var(--on-surface);border-radius:12px;padding:14px 16px;
 font:13px/1.6 "Roboto Mono",ui-monospace,monospace;white-space:pre;overflow:auto;margin:14px 0}
kbd{background:var(--surface-variant);border-radius:6px;padding:1px 6px;font:13px "Roboto Mono",monospace}
.btn{width:100%;margin-top:10px;padding:16px;border:0;border-radius:100px;font:500 15px Roboto,sans-serif;
 letter-spacing:.3px;background:var(--primary);color:var(--on-primary);cursor:pointer;transition:filter .15s,box-shadow .15s}
.btn:hover{filter:brightness(1.06);box-shadow:0 2px 8px rgba(103,80,164,.4)}
.btn:active{filter:brightness(.95)}
a{color:var(--primary);text-decoration:none;font-weight:500}
.gen{background:none;border:0;color:var(--primary);font:500 13px Roboto;cursor:pointer;padding:6px 4px}
.toggle{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:20px 2px}
.tlabel{font-weight:500} .thint{font-size:12.5px;color:var(--on-surface-variant)}
.sw{position:relative;display:inline-block;width:52px;height:32px;flex:none}
.sw input{opacity:0;width:0;height:0}
.sw span{position:absolute;inset:0;background:var(--surface-variant);border:2px solid var(--outline);
 border-radius:100px;transition:.2s;cursor:pointer}
.sw span:before{content:"";position:absolute;width:16px;height:16px;left:6px;top:6px;
 background:var(--outline);border-radius:50%;transition:.2s}
.sw input:checked+span{background:var(--primary);border-color:var(--primary)}
.sw input:checked+span:before{transform:translateX(20px);width:24px;height:24px;left:4px;top:2px;background:var(--on-primary)}
"""

_FORM = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Claude wrapper · setup</title><style>{style}</style></head><body><div class=wrap>
<div class=bar><div class=logo>✦</div><h1>Claude → OpenAI wrapper</h1></div>
<div class=card>
 <span class=chip>{chip}</span>
 <form method=post action="/setup">
  <div class=field>
   <input id=wk name=wrapper_key value="{suggested}" required autocomplete=off>
   <label for=wk>Gateway API key</label>
  </div>
  <div class=hint>You invent this. Your app sends it as the API key. <button type=button class=gen onclick="g()">Generate</button></div>

  <div class=field>
   <input id=tok name=oauth_token placeholder="paste token" autocomplete=off {token_req}>
   <label for=tok>Subscription token</label>
  </div>
  <div class=hint>On your machine (needs a browser), run — then paste the printed token above:</div>
  <div class=code>unset ANTHROPIC_API_KEY
claude setup-token</div>

  <div class=field>
   <input id=md name=model value="{model}">
   <label for=md>Default model</label>
  </div>

  <div class=toggle>
   <div><div class=tlabel>Single-turn LLM mode</div>
    <div class=thint>One prompt → one response. No agent loop.</div></div>
   <label class=sw><input type=checkbox name=single_turn value=1 {single_checked}><span></span></label>
  </div>
  <div class=toggle>
   <div><div class=tlabel>Disable thinking</div>
    <div class=thint>Skip extended reasoning before the answer.</div></div>
   <label class=sw><input type=checkbox name=disable_thinking value=1 {think_checked}><span></span></label>
  </div>

  <div class=field>
   <input id=tmt name=tool_max_turns type=number min=2 max=20 value="{tool_max_turns}">
   <label for=tmt>Tool / structured turn limit</label>
  </div>
  <div class=hint>Ceiling for tool &amp; structured requests (they need a few internal turns). Lower = less agentic wandering.</div>

  <div class=field>
   <select id=ll name=log_level>{log_options}</select>
   <label for=ll>Log level</label>
  </div>
  <div class=hint>DEBUG shows prompts &amp; per-block traces; INFO shows request/result/usage.</div>

  <button class=btn type=submit>Save configuration</button>
 </form>
</div></div>
<script>function g(){{const h='0123456789abcdef';let s='sk-local-';for(let i=0;i<32;i++)s+=h[Math.floor(Math.random()*16)];document.getElementById('wk').value=s;}}</script>
</body></html>"""

_LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR"]


@router.get("/setup", response_class=HTMLResponse)
async def setup_form():
    configured = config.is_configured()
    current_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_options = "".join(
        f'<option value="{lv}"{" selected" if lv == current_level else ""}>{lv}</option>'
        for lv in _LOG_LEVELS
    )
    return _FORM.format(
        style=_STYLE,
        chip="✓ Configured — edit anytime" if configured else "First-run setup",
        suggested=config.API_KEY if configured else "sk-local-" + secrets.token_hex(16),
        token_req="" if configured else "required",
        model=config.DEFAULT_MODEL,
        single_checked="checked" if config.SINGLE_TURN else "",
        think_checked="checked" if config.DISABLE_THINKING else "",
        tool_max_turns=config.TOOL_MAX_TURNS,
        log_options=log_options,
    )


@router.post("/setup", response_class=HTMLResponse)
async def setup_save(
    request: Request,
    wrapper_key: str = Form(...),
    oauth_token: str = Form(""),
    model: str = Form("claude-opus-4-8"),
    single_turn: str = Form("0"),      # checkbox: present ("1") only when checked
    disable_thinking: str = Form("0"),
    tool_max_turns: str = Form("5"),
    log_level: str = Form("INFO"),
):
    if not oauth_token and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        raise HTTPException(status_code=400, detail="Subscription token required on first setup")
    try:
        turns = max(2, min(20, int(tool_max_turns)))
    except ValueError:
        turns = 5
    level = log_level.upper() if log_level.upper() in _LOG_LEVELS else "INFO"
    _persist({
        "WRAPPER_API_KEY": wrapper_key,
        "CLAUDE_CODE_OAUTH_TOKEN": oauth_token,
        "CLAUDE_MODEL": model,
        "SINGLE_TURN": "1" if single_turn == "1" else "0",
        "DISABLE_THINKING": "1" if disable_thinking == "1" else "0",
        "TOOL_MAX_TURNS": str(turns),
        "LOG_LEVEL": level,
    })
    log_reconfigure()  # apply new log level live
    return _SAVED.format(style=_STYLE, key=wrapper_key, model=model)


_SAVED = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Saved</title><style>{style}</style></head><body><div class=wrap>
<div class=bar><div class=logo>✓</div><h1>Configuration saved</h1></div>
<div class=card>
 <span class=chip>Ready</span>
 <div class=hint>Base URL</div><div class=code>http://&lt;host&gt;:PORT/v1</div>
 <div class=hint>API key (use this in your app)</div><div class=code>{key}</div>
 <div class=hint>Quick test</div>
 <div class=code>curl http://localhost:PORT/v1/chat/completions \\
  -H 'Authorization: Bearer {key}' \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}'</div>
 <p style="margin-top:20px"><a href="/setup">← Back to setup</a></p>
</div></div></body></html>"""
