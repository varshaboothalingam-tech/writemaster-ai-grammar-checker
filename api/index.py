"""Vercel serverless entry point.

The dependency-light AI-first pipeline (pipeline/rule_detector + aggregator +
Gemini analyze/verify) contains NO spaCy / NLTK imports, so it runs inside a
Vercel Python function. The heavy deterministic engines (spaCy v4) are not
importable here — Gemini provides the grammar brain instead.

Endpoints:
    POST /api/check   -> {success, original_text, corrected_text, errors, ...}
    GET  /            -> branded HTML frontend (this file embeds the page)

Set GEMINI_API_KEY as a Vercel environment variable for full AI analysis.
Without a key the function degrades to the conservative offline rule path.
"""

import json
import os
import time
import traceback

from flask import Flask, Response, request

from pipeline.ai_core import check_ai_text
from pipeline.ai_analyzer import ai_key_configured, analyze, build_prompt, _provider_callable
try:
    from ai_validator import get_validator, provider_status
except Exception:  # pragma: no cover
    get_validator = provider_status = None

app = Flask(__name__)

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WriteMaster AI — Grammar Checker</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');
:root {
  --wine:#722f37; --wine-dark:#4a1a20; --wine-light:#a45a63;
  --blush:#e8d5da; --cream:#faf3f0; --gold:#c9a564;
  --ink:#2c1216; --panel:rgba(255,252,250,.78);
  --good:#2e7d5b; --bad:#b5446e;
}
* { box-sizing:border-box; }
html,body { height:100%; }
body {
  margin:0; min-height:100vh; color:var(--ink);
  font-family:'Outfit',system-ui,sans-serif;
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(169,90,99,.38), transparent 60%),
    radial-gradient(900px 500px at 110% 0%, rgba(201,165,100,.32), transparent 55%),
    radial-gradient(800px 600px at 50% 120%, rgba(114,47,55,.20), transparent 60%),
    linear-gradient(160deg, #faf3f0 0%, #f3e3e4 45%, #ead6da 100%);
  background-attachment:fixed; overflow-x:hidden;
}
.grain { position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.45;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3CfeColorMatrix values='0 0 0 0 .9 0 0 0 0 .75 0 0 0 0 .7 0 0 0 .05 0'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)'/%3E%3C/svg%3E"); }
.wrap { position:relative; z-index:1; max-width:900px; margin:0 auto; padding:56px 20px 80px; }
.hero { text-align:center; animation:rise .9s ease both; }
.crest { width:80px; height:80px; margin:0 auto 20px; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, var(--wine-light), var(--wine-dark));
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 12px 34px rgba(114,47,55,.38), inset 0 0 0 2px rgba(255,255,255,.28);
  animation:float 5s ease-in-out infinite; }
.crest svg { width:40px; height:40px; }
h1 { font-family:'Cormorant Garamond',serif; font-weight:700; font-size:clamp(2.5rem,6.5vw,3.8rem);
  margin:0; letter-spacing:.5px;
  background:linear-gradient(120deg, var(--wine-dark), var(--wine) 45%, var(--wine-light));
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:glow 4s ease-in-out infinite; }
.tag { color:var(--wine); font-style:italic; font-family:'Cormorant Garamond',serif;
  font-size:1.15rem; display:flex; align-items:center; justify-content:center; gap:14px; margin:12px 0 0; }
.tag::before,.tag::after { content:""; height:1px; width:60px;
  background:linear-gradient(90deg, transparent, var(--gold)); }
.tag::after { background:linear-gradient(90deg, var(--gold), transparent); }
.rule { width:190px; height:3px; margin:28px auto 0; border-radius:2px;
  background:linear-gradient(90deg, transparent, var(--gold), transparent); }
.panel { position:relative; margin-top:38px; padding:28px; border-radius:24px;
  background:var(--panel); backdrop-filter:blur(14px);
  border:1px solid rgba(114,47,55,.18);
  box-shadow:0 26px 64px rgba(74,26,32,.16), inset 0 1px 0 rgba(255,255,255,.85);
  animation:rise .9s .15s ease both; }
.panel::before { content:""; position:absolute; inset:0; border-radius:24px; padding:1px;
  background:linear-gradient(135deg, rgba(201,165,100,.7), transparent 30%, transparent 70%, rgba(114,47,55,.55));
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none; }
label { display:block; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--wine); font-weight:600; margin-bottom:10px; }
textarea { width:100%; min-height:150px; resize:vertical; border-radius:14px;
  border:1px solid rgba(114,47,55,.26); background:rgba(255,255,255,.78);
  padding:15px 16px; font-size:1.05rem; color:var(--ink); line-height:1.6;
  font-family:'Outfit',sans-serif; transition:border-color .3s, box-shadow .3s; }
textarea:focus { outline:none; border-color:var(--wine); box-shadow:0 0 0 4px rgba(169,90,99,.16); }
.samples { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.samples button { background:transparent; color:var(--wine); border:1px solid rgba(114,47,55,.32);
  border-radius:99px; padding:7px 15px; font-size:.82rem; cursor:pointer; font-family:'Outfit',sans-serif;
  transition:all .25s; }
.samples button:hover { background:var(--wine); color:var(--cream); transform:translateY(-2px);
  box-shadow:0 8px 18px rgba(114,47,55,.3); }
.actions { margin-top:20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
.btn { position:relative; overflow:hidden; border:none; cursor:pointer; border-radius:16px;
  padding:14px 32px; font-size:1.02rem; font-weight:600; font-family:'Outfit',sans-serif;
  color:var(--cream);
  background:linear-gradient(135deg, var(--wine-light), var(--wine) 55%, var(--wine-dark));
  box-shadow:0 12px 28px rgba(114,47,55,.36); transition:transform .2s, box-shadow .25s; }
.btn::after { content:""; position:absolute; top:0; left:-80%; width:50%; height:100%;
  background:linear-gradient(120deg, transparent, rgba(255,255,255,.38), transparent);
  transform:skewX(-20deg); }
.btn:hover { transform:translateY(-2px); box-shadow:0 16px 36px rgba(114,47,55,.44); }
.btn:hover::after { animation:sheen 1s ease; }
.btn:disabled { opacity:.6; cursor:progress; transform:none; }
.spin { display:none; width:18px; height:18px; border:3px solid rgba(114,47,55,.2);
  border-top-color:var(--wine); border-radius:50%; animation:rot .7s linear infinite; }
.out { margin-top:30px; display:flex; flex-direction:column; gap:18px; }
.card { padding:24px; border-radius:20px; background:var(--panel); backdrop-filter:blur(12px);
  border:1px solid rgba(114,47,55,.16); box-shadow:0 16px 40px rgba(74,26,32,.12);
  animation:rise .5s ease both; }
.card h3 { margin:0 0 14px; font-size:.8rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--wine); font-weight:700; display:flex; align-items:center; gap:9px; }
.dot { width:8px; height:8px; border-radius:50%; background:var(--gold); box-shadow:0 0 0 4px rgba(201,165,100,.24); }
.corrected { position:relative; background:rgba(255,255,255,.78); border:1px solid rgba(114,47,55,.16);
  border-radius:14px; padding:16px; font-size:1.18rem; line-height:1.9; white-space:pre-wrap;
  font-family:'Cormorant Garamond',serif; font-weight:600; }
.diff ins { background:rgba(46,125,91,.16); color:var(--good); text-decoration:none;
  border-bottom:2px solid var(--good); padding:0 4px; border-radius:4px; animation:pop .4s ease both; }
.diff del { background:rgba(181,68,110,.16); color:var(--bad); text-decoration:none;
  border-radius:4px; padding:0 4px; }
.issues { list-style:none; margin:0; padding:0; }
.issues li { display:grid; grid-template-columns:120px 26px 120px 1fr; gap:10px;
  align-items:center; padding:14px 0; border-bottom:1px dashed rgba(114,47,55,.2); }
.issues li:last-child { border:none; }
.iw { color:#5a1a24; font-weight:600; text-align:right; text-decoration:line-through;
  text-decoration-color:rgba(181,68,110,.6); }
.arrow { color:var(--gold); font-weight:600; text-align:center; animation:pulse 1.6s ease infinite; }
.ic { color:var(--good); font-weight:700; }
.it { color:var(--wine); font-size:.88rem; display:flex; align-items:center; gap:6px; }
.chip { background:rgba(114,47,55,.1); border-radius:8px; padding:3px 9px; font-size:.74rem;
  letter-spacing:.05em; text-transform:lowercase; font-weight:600; color:var(--wine);
  border:1px solid rgba(114,47,55,.18); }
.why { color:#6b4a52; font-size:.85rem; margin-top:4px; grid-column:1/-1; }
.badge { display:inline-flex; align-items:center; gap:8px; border-radius:99px; padding:9px 18px;
  font-size:.84rem; font-weight:600; }
.badge.err { background:rgba(181,68,110,.14); color:var(--bad); }
.badge.ok { background:rgba(46,125,91,.14); color:var(--good); }
.badge.warn { background:rgba(160,120,35,.14); color:#8a6a1f; }
.meta { color:#7d5b63; font-size:.8rem; margin-top:14px; display:flex; gap:14px; flex-wrap:wrap; }
.meta b { color:var(--wine); }
.footer { text-align:center; color:#8a6a71; font-size:.9rem; margin-top:44px;
  font-family:'Cormorant Garamond',serif; font-style:italic; animation:rise 1s .5s ease both; }
@keyframes rise { from { opacity:0; transform:translateY(28px); } to { opacity:1; transform:none; } }
@keyframes float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-9px); } }
@keyframes glow { 0%,100% { filter:drop-shadow(0 0 0 rgba(114,47,55,0)); }
  50% { filter:drop-shadow(0 4px 22px rgba(169,90,99,.45)); } }
@keyframes rot { to { transform:rotate(360deg); } }
@keyframes sheen { to { left:130%; } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }
@keyframes pop { from { transform:scale(.6); opacity:0; } to { transform:scale(1); opacity:1; } }
@media (max-width:600px){
  .issues li { grid-template-columns:1fr; text-align:left; }
  .iw,.arrow { text-align:left; }
}
</style>
</head>
<body>
<div class="grain"></div>
<div class="wrap">
  <div class="hero">
    <div class="crest">
      <svg viewBox="0 0 24 24" fill="none" stroke="#faf3f0" stroke-width="1.6">
        <path d="M12 3l2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 8.7l5.4-.8L12 3z"/>
      </svg>
    </div>
    <h1>WriteMaster AI</h1>
    <p class="tag">refined grammar, poured slowly</p>
    <div class="rule"></div>
  </div>
  <div class="panel">
    <label for="text">Your text</label>
    <textarea id="text" placeholder="Type or paste a sentence…  e.g.  She go to school everyday and she dont like it"></textarea>
    <div class="samples">
      <button type="button" onclick="setSample(0)">She go to school…</button>
      <button type="button" onclick="setSample(1)">I have a good time yesterday.</button>
      <button type="button" onclick="setSample(2)">He dont like coffee</button>
      <button type="button" onclick="setSample(3)">The men is walking fast</button>
    </div>
    <div class="actions">
      <button class="btn" id="btn" onclick="check()">&#10005; Check grammar</button>
      <span class="spin" id="spin"></span>
    </div>
  </div>
  <div class="out" id="out" style="display:none"></div>
  <div class="footer">&#8226; verified by Google Gemini &#8226; every word mellowed to perfection &#8226;</div>
</div>
<script>
const samples = [
  "She go to school everyday and she dont like it",
  "I have a good time yesterday.",
  "He dont like coffee",
  "The men is walking fast"
];
function setSample(i){ document.getElementById('text').value = samples[i]; }
function esc(s){ return String(s).replace(/[&<>"']/g, function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
async function check(){
  const t = document.getElementById('text').value.trim();
  const btn = document.getElementById('btn'), spin = document.getElementById('spin'),
        out = document.getElementById('out');
  if(!t){ document.getElementById('text').focus(); return; }
  btn.disabled = true; spin.style.display = 'inline-block';
  out.style.display = 'none';
  try{
    const r = await fetch('/api/check', { method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: t }) });
    const j = await r.json();
    render(j);
  }catch(e){
    out.style.display = 'block';
    out.innerHTML = '<div class="card"><h3><span class="dot"></span>Sip interrupted</h3><p>'+
      esc(e.message)+'</p></div>';
  }finally{
    btn.disabled = false; spin.style.display = 'none';
  }
}
function render(j){
  const out = document.getElementById('out');
  out.style.display = 'flex';
  if(!j.success){
    out.innerHTML = '<div class="card"><h3><span class="dot"></span>Something went wrong</h3><p>'+
      esc(j.message || 'unknown error') + '</p></div>';
    return;
  }
  const errors = j.errors || [];
  const hasChanges = !!j.corrected_text && j.corrected_text !== j.original_text;
  let html = '<div class="card">' +
    '<h3><span class="dot"></span>Verdict</h3>' +
    '<span class="badge '+(errors.length?'err':'ok')+'">' +
    (errors.length ? '&there4; '+errors.length+' issue'+(errors.length>1?'s':'')+' found'
                   : '&check; no errors detected') + '</span></div>';
  if(hasChanges){
    html += '<div class="card"><h3><span class="dot"></span>Corrected text</h3>' +
      '<div class="corrected diff">'+esc(j.corrected_text)+'</div></div>';
  }
if(errors.length){
      let items = '';
      for(const e of errors){
        const tag = e.consensus ? esc(e.consensus.toLowerCase()) : '';
        items += '<li>' +
          '<span class="iw">'+esc(e.wrong)+'</span>' +
          '<span class="arrow">&rarr;</span>' +
          '<span class="ic">'+esc(e.correct)+'</span>' +
          '<span class="it"><span class="chip">'+esc(e.type||'grammar')+
            (tag ? ' &middot; '+tag : '')+'</span></span>' +
          '<span class="why">'+esc(e.message||e.explanation||'')+'</span></li>';
      }
      html += '<div class="card"><h3><span class="dot"></span>Issues</h3><ul class="issues">'+items+'</ul></div>';
    }
    const consensus = j.consensus||{}, metaChips = [];
    for(const k of ['agreed','ai_only','local_only']){
      if((consensus[k]||0) > 0) metaChips.push(k.replace('_','-')+': <b>'+consensus[k]+'</b>');
    }
    html += '<div class="card"><div class="meta">' +
      '<span>brain: <b>'+(j.ai_used?'Google Gemini':'local rules')+'</b></span>' +
      '<span>status: <b>'+esc(j.grammar_status||'')+'</b></span>' +
      '<span>took: <b>'+j.processing_time_ms+' ms</b></span>' +
      (metaChips.length ? '<span>' + metaChips.join(' &middot; ') + '</span>' : '') +
      '</div></div>';
  out.innerHTML = html;
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def root():
    return Response(_INDEX_HTML, status=200, mimetype="text/html")


@app.route("/api/check", methods=["POST"])
def api_check():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "") or "").strip()
    if not text:
        return Response(
            json.dumps({"success": False, "message": "empty text",
                        "issues": [], "errors": [], "corrected_text": ""}),
            status=400, mimetype="application/json")
    try:
        result = check_ai_text(text, use_ai=payload.get("use_ai", True)
                               if ai_key_configured() else False)
        meta = result.get("meta", {})
        verification = meta.get("verification", {}) or {}
        return Response(
            json.dumps({
                "success": True,
                "original_text": result["original_text"],
                "corrected_text": result["corrected_text"],
                "errors": result["errors"],
                "grammar_status": result["grammar_status"],
                "ai_used": meta.get("ai_used", False),
                "pipeline": meta.get("pipeline", "master"),
                "schema": result.get("schema", "v2_error_object"),
                "quality": result.get("quality", {}),
                "consensus": result.get("consensus", {}),
                "verification": verification.get("decision"),
                "processing_time_ms": result["processing_time_ms"],
            }, ensure_ascii=False),
            status=200, mimetype="application/json")
    except Exception as exc:  # pragma: no cover - defensive
        return Response(
            json.dumps({"success": False, "message": str(exc),
                        "issues": [], "errors": [], "corrected_text": text}),
            status=500, mimetype="application/json")


@app.route("/api/diag", methods=["GET"])
def api_diag():
    info = {
        "ai_configured": ai_key_configured(),
        "gemini_key_present": bool(os.environ.get("GEMINI_API_KEY")),
        "ai_provider_env": os.environ.get("AI_PROVIDER", ""),
        "ai_model_env": os.environ.get("AI_MODEL", ""),
    }
    v = get_validator() if get_validator else None
    if v is not None:
        info["provider"] = v.provider
        info["model"] = v.model
        info["providers"] = v.providers
        info["is_enabled"] = v.is_enabled()
        try:
            info["available"] = v.available()
        except Exception as exc:
            info["available"] = f"error: {exc}"
    if provider_status is not None:
        try:
            info["provider_status"] = provider_status()
        except Exception as exc:
            info["provider_status"] = f"error: {exc}"
    try:
        call = _provider_callable()
        info["provider_callable"] = call is not None
        if call is not None and os.environ.get("DIAG_DEEP", "").strip() in ("1", "true"):
            from pipeline.rule_detector import detect_rules
            from pipeline.aggregator import aggregate, relocate_candidates
            cands = relocate_candidates(aggregate(detect_rules("She go to school everyday and she dont like it."), []),
                                        "She go to school everyday and she dont like it.")
            try:
                prompt = build_prompt("She go to school everyday and she dont like it.", cands)
                raw = call(prompt)
                info["raw_len"] = len(raw or "")
                info["raw_full"] = str(raw)[:2000]
            except Exception as exc:
                info["call_exception"] = f"{type(exc).__name__}: {exc}"
                info["call_traceback"] = "\n".join(traceback.format_exc().splitlines()[-6:])
            try:
                res = analyze("She go to school everyday and she dont like it.", cands, raw_call=call)
                info["analyze_result_type"] = type(res).__name__
                if res is None:
                    info["analyze_none_hint"] = "analyze returned None"
                else:
                    info["analyze_errors_count"] = len(res.get("errors") or [])
                    info["analyze_status"] = res.get("grammar_status")
            except Exception as exc:
                info["analyze_exception"] = f"{type(exc).__name__}: {exc}"
                traceback_snippet = "\n".join(traceback.format_exc().splitlines()[-8:])
                info["analyze_traceback"] = traceback_snippet
    except Exception as exc:
        info["diag_call_error"] = f"{type(exc).__name__}: {exc}"
    return Response(json.dumps(info, ensure_ascii=False, default=str),
                    status=200, mimetype="application/json")


# Vercel's @vercel/python build serves this WSGI app directly.
wsgi_app = app