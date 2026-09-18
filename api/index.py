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
import threading
import time
import traceback

from flask import Flask, Response, request

from pipeline.ai_core import check_ai_text
from pipeline.ai_analyzer import ai_key_configured, analyze, build_prompt, _provider_callable, _extract_json
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#edf1f7; --card:#ffffff; --ink:#1e293b; --muted:#637083; --line:#e3e8f0;
  --acc:#00a272; --acc-d:#00835c; --acc-l:#e6f7f1; --bad:#d23b3b; --bad-l:#feeceb;
  --good:#178a4f; --warn:#b78a2f; --wine:#1e293b;
  --shadow:0 8px 24px rgba(20,40,70,.07);
}
html,body{height:100%}
body{background:
  radial-gradient(1200px 500px at 85% -10%, rgba(0,162,114,.10), transparent 60%),
  radial-gradient(900px 420px at -10% 0%, rgba(59,130,246,.08), transparent 55%),
  var(--bg);
  color:var(--ink);font-family:'Outfit',-apple-system,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased}
.topbar{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;
  gap:14px;padding:10px clamp(14px,3vw,28px);background:rgba(255,255,255,.86);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.tbl{display:flex;align-items:center;gap:14px}
.logo{display:flex;align-items:center;gap:10px}
.logo-ico{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;
  background:linear-gradient(135deg,#00b884,#00835c);color:#fff;font-size:1.3rem;
  box-shadow:0 6px 14px rgba(0,162,114,.35)}
.logo-txt b{font-size:1.02rem;letter-spacing:.2px;display:block;line-height:1.1}
.logo-txt small{font-size:.66rem;color:var(--muted);letter-spacing:.6px;text-transform:uppercase}
.tools{display:flex;gap:6px;margin-left:8px}
.tool{font-size:.78rem;font-weight:600;color:var(--muted);padding:6px 12px;border-radius:999px;cursor:default}
.tool.active{background:var(--acc-l);color:var(--acc-d)}
.tbr{display:flex;align-items:center;gap:16px}
.ai-toggle{display:flex;align-items:center;gap:7px;font-size:.78rem;font-weight:600;color:var(--muted);cursor:pointer;user-select:none}
.ai-toggle input{display:none}
.ai-toggle .sw{width:38px;height:21px;border-radius:999px;background:#cbd5e1;position:relative;transition:.25s}
.ai-toggle .sw::after{content:'';position:absolute;top:2.5px;left:3px;width:16px;height:16px;border-radius:50%;background:#fff;transition:.25s;box-shadow:0 1px 3px rgba(0,0,0,.25)}
.ai-toggle input:checked + .sw{background:var(--acc)}
.ai-toggle input:checked + .sw::after{left:19px}
.scorepill{display:flex;align-items:baseline;background:#fff;border:1px solid var(--line);padding:5px 12px;border-radius:999px;font-size:.72rem;color:var(--muted);box-shadow:var(--shadow)}
.sc-num{font-size:1.05rem;font-weight:800;color:var(--acc-d);margin-right:3px}
.stage{display:grid;grid-template-columns:minmax(0,11fr) minmax(0,13fr);gap:clamp(14px,2vw,26px);
  max-width:1280px;margin:clamp(14px,2vw,26px) auto 6px;padding:0 clamp(14px,3vw,28px)}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);
  padding:clamp(14px,2vw,22px);display:flex;flex-direction:column;gap:14px;min-width:0;overflow:hidden}
.chead{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.chead h2{font-size:1.02rem;letter-spacing:.2px}
.chead p{font-size:.78rem;color:var(--muted);margin-top:2px}
.ca{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{font-family:inherit;font-weight:700;font-size:.92rem;color:#fff;border:none;cursor:pointer;
  padding:11px 20px;border-radius:12px;background:linear-gradient(135deg,#00b884,#00835c);
  box-shadow:0 10px 22px rgba(0,162,114,.32);transition:all .18s;display:inline-flex;align-items:center;gap:8px}
.btn:hover{transform:translateY(-1px);box-shadow:0 14px 28px rgba(0,162,114,.4)}
.btn:disabled{opacity:.55;cursor:not-allowed;transform:none}
.btn.small{padding:7px 13px;font-size:.8rem;border-radius:10px}
.btn.ghost,.ghostbtn{background:#fff;color:var(--acc-d);border:1px solid rgba(0,162,114,.4);box-shadow:none}
.btn.ghost:hover,.ghostbtn:hover{background:var(--acc-l);border-color:var(--acc)}
.ghostbtn{font-family:inherit;font-weight:600;font-size:.8rem;padding:7px 13px;border-radius:10px;cursor:pointer;transition:all .18s}
.btn.accepted{background:var(--acc-l);color:var(--good);box-shadow:none;pointer-events:none}
.btn-copy{font-family:inherit;font-size:.8rem;font-weight:600;color:var(--acc-d);background:#fff;
  border:1px solid rgba(0,162,114,.4);padding:7px 14px;border-radius:10px;cursor:pointer;transition:all .18s}
.btn-copy:hover{background:var(--acc-l);border-color:var(--acc)}
.btn-copy:active{transform:translateY(1px)}
.btn-copy.copied{color:#fff;background:var(--acc);border-color:var(--acc);box-shadow:0 6px 14px rgba(0,162,114,.35)}
.edbox{position:relative}
#text{width:100%;min-height:210px;resize:vertical;border:1.5px solid var(--line);border-radius:14px;
  padding:13px 15px;font-family:inherit;font-size:1rem;line-height:1.7;color:var(--ink);background:#fff;
  outline:none;transition:border-color .18s, box-shadow .18s}
#text:focus{border-color:var(--acc);box-shadow:0 0 0 4px rgba(0,162,114,.14)}
#text::placeholder{color:#9aa7b6}
.micbtn{position:absolute;right:10px;bottom:10px;background:#fff;border:1px solid var(--line);
  color:var(--muted);border-radius:999px;padding:6px 13px;font-family:inherit;font-size:.76rem;font-weight:600;
  cursor:pointer;transition:all .18s;display:inline-flex;align-items:center;gap:6px}
.micbtn:hover{border-color:var(--acc);color:var(--acc-d)}
.micbtn.rec{background:var(--bad);color:#fff;border-color:var(--bad);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(210,59,59,.4)}50%{box-shadow:0 0 0 9px rgba(210,59,59,0)}}
.meta-row{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}
.counts{font-size:.78rem;color:var(--muted)}
.counts b{color:var(--ink)}
.samples{display:flex;gap:6px;flex-wrap:wrap}
.samples button{font-size:.72rem;color:var(--muted);background:#f4f7fb;border:1px solid var(--line);
  padding:5px 11px;border-radius:999px;cursor:pointer;transition:all .18s;font-family:inherit}
.samples button:hover{color:var(--acc-d);border-color:var(--acc)}
.ofoot{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap}
.kbd-hint{font-size:.72rem;color:var(--muted)}
.spin{width:16px;height:16px;border:2.5px solid #cfe3db;border-top-color:var(--acc);border-radius:50%;
  animation:sp .8s linear infinite;display:none}
@keyframes sp{to{transform:rotate(360deg)}}
.read-bar{display:none;gap:8px;align-items:center;flex-wrap:wrap;border-top:1px dashed var(--line);padding-top:12px}
.read-bar .rv{font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.read-bar select{font-family:inherit;font-size:.78rem;padding:6px 10px;border-radius:10px;border:1px solid var(--line);max-width:200px;color:var(--ink)}
.speakbtn{font-family:inherit;font-size:.75rem;font-weight:600;color:var(--acc-d);background:var(--acc-l);
  border:1px solid rgba(0,162,114,.25);padding:6px 12px;border-radius:999px;cursor:pointer;transition:all .18s}
.speakbtn:hover{background:#d4efe6}
.speakbtn.stop{background:var(--bad-l);color:var(--bad);border-color:rgba(210,59,59,.3)}
.finalcard{background:linear-gradient(180deg,#f2fbf7,#eefaf4);
  border:1.5px solid rgba(0,162,114,.28);border-radius:16px;padding:14px 16px}
.fc-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.fc-head h3{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--acc-d);font-weight:800}
.fc-sub{font-size:.72rem;color:var(--muted)}
.fc-tog{display:flex;gap:4px;margin-left:auto}
.fc-tog button{border:1px solid rgba(0,162,114,.3);background:#fff;color:var(--acc-d);font:600 .7rem 'Outfit',sans-serif;
  padding:4px 10px;border-radius:999px;cursor:pointer;transition:all .18s}
.fc-tog button:hover{background:var(--acc-l)}
.fc-tog button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.pholder{color:#9aa7b6;text-align:center;padding:22px 10px;border:1.5px dashed var(--line);
  border-radius:14px;font-size:.88rem;background:#fafcfe}
.pholder b{display:block;color:var(--ink);font-size:.96rem;margin-bottom:3px}
.pvcard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:.95rem;
  line-height:1.75;white-space:pre-wrap;word-break:break-word}
.pvcard.clean{background:#fff;border:1.5px solid rgba(0,162,114,.3);border-left:4px solid var(--acc)}
.pvhead{display:flex;align-items:baseline;gap:8px;margin:4px 0 8px}
.pvhead h3{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink);font-weight:700}
.pvhead small{font-size:.7rem;color:var(--muted)}
.rp-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.quillbar{display:flex;align-items:center;gap:8px;background:var(--acc-l);border:1px solid rgba(0,162,114,.2);
  border-radius:12px;padding:9px 12px;margin:4px 0 12px;font-size:.82rem;color:var(--ink)}
.qb-dot{width:8px;height:8px;border-radius:50%;background:var(--acc);display:inline-block;flex:0 0 auto}
.quillbar b{color:var(--acc-d)}
.quillbar .btn{margin-left:auto}
.typetabs{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.typetabs button{border:1px solid var(--line);background:#f6f9fc;color:var(--muted);font-family:'Outfit',sans-serif;
  font-size:.74rem;font-weight:600;padding:6px 12px;cursor:pointer;border-radius:999px;transition:all .18s}
.typetabs button:hover{border-color:var(--acc);color:var(--acc-d)}
.typetabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tbadge{display:inline-block;margin-left:5px;background:rgba(30,41,59,.14);border-radius:999px;padding:0 6px;
  font-size:.66rem;line-height:1.5;color:inherit}
.typetabs button.on .tbadge{background:rgba(255,255,255,.25);color:#fff}
.group{margin-bottom:6px}
.grouptitle{display:flex;align-items:center;gap:8px;font-size:.76rem;font-weight:800;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink);margin:12px 0 8px}
.gbadge{background:var(--acc-l);color:var(--acc-d);font-size:.66rem;padding:1px 8px;border-radius:999px;font-weight:700}
.prwrap{display:flex;flex-wrap:wrap;gap:8px;padding:2px 0 4px}
.pr{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);background:#fff;padding:6px 12px;
  border-radius:999px;font-size:.86rem;font-weight:600;cursor:pointer;transition:all .16s;font-family:'Outfit',sans-serif;color:var(--ink)}
.pr:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(20,40,70,.12);border-color:var(--acc)}
.pr del{color:var(--bad);text-decoration:line-through;text-decoration-thickness:2px}
.pr ins{color:var(--good);text-decoration:none}
.prarr{color:var(--warn);font-weight:700}
.pr.fixed{opacity:.45;pointer-events:none;border-style:dashed;background:#f2f5f9}
.pr-hint{font-size:.68rem;color:#94a3b8;padding:0 2px 2px}
.summary-bar{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.summary-chip{display:inline-flex;align-items:center;gap:6px;font-size:.74rem;font-weight:600;color:var(--muted);
  background:#f6f9fc;border:1px solid var(--line);padding:4px 10px;border-radius:999px}
.sdot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.iconly{width:9px;height:9px;border-radius:50%;display:inline-block;flex:0 0 auto}
.sdot.ig-grammar,.iconly.ig-grammar{background:#3b82f6}
.sdot.ig-spelling,.iconly.ig-spelling{background:#f59e0b}
.sdot.ig-punctuation,.iconly.ig-punctuation{background:#8b5cf6}
.sdot.ig-style,.iconly.ig-style{background:#ec4899}
.sdot.ig-clarity,.iconly.ig-clarity{background:#14b8a6}
.sdot.ig-context,.iconly.ig-context{background:#f43f5e}
.sdot.ig-other,.iconly.ig-other{background:#64748b}
.status{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.76rem;color:var(--muted);margin:2px 0 6px}
.status b{color:var(--ink)}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:2px 0 6px}
.chip{font-size:.7rem;font-weight:600;color:var(--acc-d);background:var(--acc-l);border-radius:999px;padding:3px 9px}
.empty-assist{background:#f6f9fc;border:1.5px dashed var(--line);border-radius:14px;padding:18px;text-align:center;
  color:var(--muted);font-size:.9rem;line-height:1.6}
.empty-assist .big{display:block;font-weight:800;color:var(--ink);font-size:1rem;margin-bottom:4px}
.del{color:var(--bad);text-decoration:line-through;text-decoration-thickness:2px;background:rgba(210,59,59,.12);border-radius:3px}
.ins{color:var(--good);background:rgba(23,138,79,.14);border-radius:3px;font-weight:600}
.editor-corrected{margin-top:2px}
.corrected-section{border:1px solid var(--line);border-radius:16px;padding:14px;background:#fff;margin-top:4px}
.corrected-label{font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--good);margin-bottom:8px}
.corrected-box{font-size:.95rem;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.corrected-meta{font-size:.74rem;color:var(--muted);margin-top:8px}
.corrected-actions{display:flex;gap:8px;margin-top:10px}
.preview{margin-top:8px}
mark.e{background:rgba(245,158,11,.18);border-bottom:2px solid #f59e0b;border-radius:3px;padding:0 1px;cursor:pointer;color:inherit}
mark.e.grammar{border-bottom-color:#3b82f6;background:rgba(59,130,246,.12)}
mark.e.punctuation{border-bottom-color:#8b5cf6}
mark.e.style{border-bottom-color:#ec4899}
mark.e.other{border-bottom-color:#64748b}
mark.e:hover{box-shadow:0 0 0 2px rgba(245,158,11,.25)}
.tip{position:absolute;z-index:90;max-width:290px;background:#fff;border:1px solid var(--line);border-radius:12px;
  box-shadow:0 18px 44px rgba(20,40,70,.22);padding:12px 14px;font-size:.82rem;line-height:1.55;color:var(--ink);display:none}
.tip b{font-weight:800}
.tclose{position:absolute;top:6px;right:8px;border:none;background:transparent;color:var(--muted);font-size:1.15rem;cursor:pointer}
.tfix{margin-top:8px;width:100%;background:var(--acc);color:#fff;border:none;border-radius:9px;padding:8px;font-weight:700;cursor:pointer;font-family:inherit;font-size:.82rem}
.osynpop{position:fixed;z-index:120;background:#fff;border:1px solid var(--line);border-radius:16px;
  box-shadow:0 24px 60px rgba(20,40,70,.25);padding:14px 16px;width:min(340px,92vw);display:none}
.osynpop h4{font-size:.9rem;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.osynpop h4 span{color:var(--acc-d);font-weight:800}
.osynpop .wtabs{display:flex;gap:6px;margin-bottom:8px}
.osynpop .wtabs button{flex:1;border:1px solid var(--line);background:#f6f9fc;font-family:inherit;font-size:.76rem;
  font-weight:600;padding:6px;border-radius:9px;cursor:pointer;color:var(--muted);transition:all .18s}
.osynpop .wtabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.wchips{display:flex;flex-wrap:wrap;gap:6px;max-height:180px;overflow:auto;font-size:.78rem;color:var(--muted)}
.wchips button{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 10px;font-family:inherit;font-size:.78rem;cursor:pointer;transition:all .15s;color:var(--ink)}
.wchips button:hover{border-color:var(--acc);color:var(--acc-d)}
.wnote{font-size:.68rem;color:var(--muted);margin-top:8px}
.rp-head{display:flex;align-items:center;gap:9px;margin-bottom:9px;flex-wrap:wrap}
.rp-head h3{margin:0;font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;color:var(--acc-d);font-weight:800}
.rp-head small{color:var(--muted);font-size:.72rem}
.rp-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.rp-col{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px;font-size:.93rem;
  line-height:1.75;white-space:pre-wrap;word-break:break-word;min-height:80px}
.rp-col .rp-tag{display:block;font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;font-weight:800;margin-bottom:6px}
.rp-col.orig .rp-tag{color:var(--muted)}
.rp-col.rep .rp-tag{color:var(--good)}
.rp-note{font-size:.72rem;color:var(--muted);margin-top:8px}
.issue{padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:#fff;margin-bottom:6px}
.issue.done{opacity:.55}
.ibody .ih{font-size:.76rem;font-weight:700;color:var(--muted);margin-bottom:4px}
.ibody .orig{color:var(--bad)}
.ibody .gud{color:var(--good)}
.ibody .why{font-size:.8rem;color:var(--muted);line-height:1.5}
.arr{color:var(--warn);font-weight:700}
.footer{max-width:1280px;margin:6px auto 26px;padding:0 20px;text-align:center;font-size:.72rem;color:#8a97a8;letter-spacing:.05em}
.osynpop{}
@media (max-width:960px){
  .stage{grid-template-columns:1fr}
  .tools{display:none}
}
@media (max-width:520px){
  .scorepill{display:none}
  .topbar{padding:8px 12px}
  .card{padding:14px;border-radius:16px}
  #text{min-height:150px;font-size:.95rem}
  .meta-row{flex-direction:column;align-items:stretch}
  .ofoot{justify-content:space-between}
}
@media (max-width:640px){.rp-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="topbar">
  <div class="tbl">
    <div class="logo">
      <span class="logo-ico">&#9998;</span>
      <div class="logo-txt"><b>WriteMaster</b><small>AI Grammar Checker</small></div>
    </div>
    <nav class="tools"><span class="tool active">Grammar Checker</span></nav>
  </div>
  <div class="tbr">
    <label class="ai-toggle" title="Final validation by Google Gemini (needs a configured key)">
      <span>AI</span>
      <input type="checkbox" id="useAi" checked>
      <span class="sw"></span>
    </label>
    <div class="scorepill" title="Writing quality estimate"><span class="sc-num" id="scoreNum">&ndash;</span>/ 100</div>
  </div>
</header>

<main class="stage">
  <section class="card">
    <div class="chead">
      <div>
        <h2>Original text</h2>
        <p>Paste or type your text below</p>
      </div>
      <div class="ca"><button class="micbtn" id="micBtn" title="Dictate your text (speech to text)">Dictate</button></div>
    </div>
    <div class="edbox">
      <textarea id="text" items="10" spellcheck="false" placeholder="Type or paste a sentence&#8230;  e.g.  She go to school everyday and she dont like it"></textarea>
    </div>
    <div class="meta-row">
      <div class="counts"><b id="cChars">0</b> chars &middot; <b id="cWords">0</b> words &middot; <b id="cSents">0</b> sentences <span class="kbd-hint">(Ctrl+Enter)</span></div>
      <div class="samples">
        <button type="button" onclick="setSample(0)">She go to school&#8230;</button>
        <button type="button" onclick="setSample(1)">I have a good time yesterday.</button>
        <button type="button" onclick="setSample(2)">He dont like coffee</button>
        <button type="button" onclick="setSample(3)">The men is walking fast</button>
      </div>
    </div>
    <div class="ofoot">
      <span class="spin" id="spin"></span>
      <button class="btn" id="btn" onclick="check()">&#10005; Check grammar</button>
    </div>
    <div class="read-bar" id="speechBar" style="display:none">
      <span class="rv">Read aloud</span>
      <select id="voiceSel" title="Choose a voice"></select>
      <button class="speakbtn" id="btnSpo" onclick="speak('orig')">Speak original</button>
      <button class="speakbtn" id="btnSpc" onclick="speak('corr')">Speak corrected</button>
      <button class="speakbtn" id="btnStop" onclick="stopSpeech()" style="display:none">Stop</button>
    </div>
    <div id="editorCorrected" class="editor-corrected"></div>
    <div id="previewPane" class="preview"></div>
  </section>

  <section class="card">
    <div class="chead">
      <div>
        <h2>Corrections</h2>
        <p id="resSub">Your fixed text will appear here</p>
      </div>
      <div class="ca">
        <button class="btn small" id="btnRephrase" style="display:none" onclick="rephrase()">&#10024; Rephrase</button>
        <button class="ghostbtn" id="btnReset" style="display:none">Reset</button>
      </div>
    </div>
    <div id="finalCard" class="finalcard" style="display:none"></div>
    <div class="typetabs" id="typeTabs" style="display:none"></div>
    <div id="assistBody"><div class="pholder"><b>Ready when you are</b>Paste a sentence and click <b>Check grammar</b>.</div></div>
    <div id="rephrasePane" class="preview"></div>
  </section>
</main>

<div class="osynpop" id="wpop">
  <h4><span id="wpopWord"></span> &mdash; word tools</h4>
  <div class="wtabs">
    <button type="button" id="tabSyn" class="on" onclick="wtab('syn')">Synonyms</button>
    <button type="button" id="tabAnt" onclick="wtab('ant')">Antonyms</button>
  </div>
  <div class="wchips" id="wchips">Double-click a word above to see synonyms &amp; antonyms.</div>
  <div class="wnote">Synonyms &amp; antonyms by Datamuse &middot; click a word to replace it</div>
</div>

<footer class="footer">&#8226; verified by Google Gemini &#8226; every word mellowed to perfection &#8226;</footer>
</body>
<script>
const samples = [
  "She go to school everyday and she dont like it",
  "I have a good time yesterday.",
  "He dont like coffee",
  "The men is walking fast"
];
let state = { errors: [], applied: {}, lastResult: null, cards: null, typeCards: null, typeCounts: null, metaHtml: '', rephrased: '' };
function setSample(i){ document.getElementById('text').value = samples[i]; updateCounts(); }
function esc(s){ return String(s).replace(/[&<>"']/g, function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
function grpOf(t){ t = (t||'other').toLowerCase();
  return {spelling:'Spelling',grammar:'Grammar',punctuation:'Punctuation',style:'Style',clarity:'Clarity',context:'Context',semantic:'Style',tense:'Grammar'}[t] || 'Other'; }
function igOf(t){ t = (t||'other').toLowerCase();
  return ['spelling'].includes(t) ? 'spelling' :
         ['grammar','tense'].includes(t) ? 'grammar' :
         ['punctuation'].includes(t) ? 'punctuation' :
         ['style','semantic','clarity'].includes(t) ? 'style' : 'other'; }
function updateCounts(){
  const t = document.getElementById('text').value;
  document.getElementById('cChars').textContent = t.length;
  document.getElementById('cWords').textContent = (t.trim()? t.trim().split(/\s+/).length : 0);
  const s = (t.match(/[.!?…]+(\s|$)/g)||[]).length;
  document.getElementById('cSents').textContent = Math.max(s, t.trim()? 1:0);
}
async function check(){
  const t = document.getElementById('text').value.trim();
  const btn = document.getElementById('btn'), spin = document.getElementById('spin');
  if(!t){ document.getElementById('text').focus(); return; }
  btn.disabled = true; spin.style.display = 'inline-block';
  try{
    const r = await fetch('/api/check', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ text: t, use_ai: document.getElementById('useAi').checked }) });
    const j = await r.json();
    render(j);
  }catch(e){
    document.getElementById('assistBody').innerHTML =
      '<div class="empty-assist"><span class="big">Sip interrupted</span>' + esc(e.message) + '</div>';
  }finally{
    btn.disabled = false; spin.style.display = 'none';
  }
}
function wordGuess(t){ const w = t.trim().split(/\s+/).length; return w; }
function render(j){
  const body = document.getElementById('assistBody');
  document.getElementById('btnReset').style.display = 'inline-block';
  const errors = (j.errors || []).slice().sort(function(a,b){ return a.start - b.start; });
  state.errors = errors; state.applied = {}; state.original = j.original_text; state.good = j.corrected_text;
  if(!j.success){
    body.innerHTML = '<div class="empty-assist"><span class="big">Something went wrong</span>'+
      esc(j.message || 'unknown error') + '</div>';
    setScore(null); return;
  }
  const q = j.quality||{}, conf = q.confidence;
  setScore(conf == null ? null : Math.round(conf * 100));

  let meta = '<div class="status">' +
    '<span>brain: <b>'+esc(j.ai_used?'Google Gemini':'local rules')+'</b></span>' +
    '<span>status: <b>'+esc(j.grammar_status||'')+'</b></span>' +
    '<span>took: <b>'+j.processing_time_ms+' ms</b></span></div>';
  const cons = j.consensus||{}, chips = [];
  for(const k of ['agreed','ai_only','local_only']){
    if((cons[k]||0) > 0) chips.push(k.replace('_','-') + ' ' + cons[k]);
  }
  if(chips.length) meta += '<div class="chips">'+ chips.map(function(c){ return '<span class="chip">'+c+'</span>'; }).join('') +'</div>';
  meta += summaryBarHtml(errors);

  if(!errors.length){
    body.innerHTML = meta + '<div class="empty-assist"><span class="big">No issues found</span>' +
      (j.ai_used ? 'Gemini read it twice and found nothing to pour away.' :
                  'Your text reads clean.' ) + '</div>';
    var ecN=document.getElementById('editorCorrected');
    if(ecN) ecN.innerHTML='';
    var fcN=document.getElementById('finalCard');
    if(fcN) fcN.style.display='none';
    var ttN=document.getElementById('typeTabs');
    if(ttN) ttN.style.display='none';
    return;
  }
  const groups = {};
  errors.forEach(function(e, i){ ++(groups[grpOf(e.type)] = groups[grpOf(e.type)] || {}).n ||
    (groups[grpOf(e.type)] = { n:1, list:[] }).list; groups[grpOf(e.type)].list.push({e:e, i:i}); });
  let html = meta + '<div class="quillbar"><span class="qb-dot"></span><b>' + errors.length +
    '</b> corrections found' +
    '<button class="btn small ghost" onclick="applyAll()">&#10003; Apply all</button></div>';
  const typeCards = {};
  const typeCounts = {};
  Object.keys(groups).forEach(function(g){
    const gd = groups[g];
    let ghtml = '<div class="group"><div class="grouptitle"><span class="iconly ig-'+igOf(gd.list[0].e.type)+'"></span>' +
      g + ' <span class="gbadge">'+gd.n+'</span></div><div class="prwrap">';
    gd.list.forEach(function(item){ ghtml += pairHtml(item.e, item.i); });
    ghtml += '</div><div class="pr-hint">Tap a pair to apply it &middot; hover for the reason</div></div>';
    typeCards[g] = ghtml; typeCounts[g] = gd.n;
    html += ghtml;
  });
  state.metaHtml = meta;
  state.typeCards = typeCards;
  state.typeCounts = typeCounts;
  state.cards = {
    all: html,
    orig: '<div class="pvcard">'+esc(j.original_text)+'</div>',
    corr: (j.corrected_text && j.corrected_text!==j.original_text)
        ? '<div class="pvcard diff">'+(diffHtml(j)||esc(j.corrected_text))+'</div>'
        : '<div class="empty-assist"><span class="big">No changes</span>The text is already correct.</div>'
  };
  body.innerHTML = html;
  buildTypeTabs();
  renderFinalCard(j);
  for(const i in state.applied) applyMark(i, !!state.applied[i]);
  var ec=document.getElementById('editorCorrected');
  if(ec) ec.innerHTML = correctedTextHtml(j);
  state.lastResult = j;
  var at2=document.getElementById('typeTabs');
  if(at2) at2.style.display='flex';
  document.getElementById('btnRephrase').style.display='inline-block';
  var cs=document.getElementById('correctedSection');
  if(cs) setTimeout(function(){cs.scrollIntoView({behavior:'smooth',block:'nearest'});},150);
}
function buildTypeTabs(){
  const wrap=document.getElementById('typeTabs');
  if(!wrap) return;
  if(!state.typeCards) return;
  const order=['Grammar','Spelling','Punctuation','Style','Clarity','Context','Other'];
  const keys=order.filter(function(g){ return state.typeCards[g]; });
  Object.keys(state.typeCards).forEach(function(g){
    if(order.indexOf(g)<0) keys.push(g);
  });
  let h='<button type="button" data-t="__all" class="on" onclick="typeFilter(\'__all\')">All corrections</button>';
  keys.forEach(function(g){
    h+='<button type="button" data-t="'+esc(g)+'" onclick="typeFilter(\''+esc(g)+'\')">'+esc(g)+
       ' <span class="tbadge">'+state.typeCounts[g]+'</span></button>';
  });
  wrap.innerHTML=h;
  wrap.style.display='flex';
}
function typeFilter(k){
  const wrap=document.getElementById('typeTabs');
  if(wrap){ Array.prototype.forEach.call(wrap.children, function(b){ b.classList.toggle('on', b.getAttribute('data-t')===k); }); }
  const body=document.getElementById('assistBody');
  if(!body || !state.cards) return;
  let h=state.metaHtml||'';
  if(k==='__all'){ h+=state.cards.all; }
  else if(state.typeCards[k]){ h+=state.typeCards[k]; }
  body.innerHTML=h;
  for(const i in state.applied) applyMark(i, !!state.applied[i]);
}
function renderFinalCard(j, mode){
  const fc=document.getElementById('finalCard');
  if(!fc) return;
  mode = mode || 'clean';
  fc.style.display='block';
  const changed=!!(j.corrected_text && j.corrected_text!==j.original_text);
  const body = mode==='diff' ? (diffHtml(j)||esc(j.corrected_text||''))
                            : esc(j.corrected_text||j.original_text||'');
  fc.innerHTML='<div class="fc-head"><h3>Corrected</h3><span class="fc-sub">'+
    (changed?'final version':'no changes needed')+'</span>'+
    '<div class="fc-tog">'+
      '<button type="button" class="'+(mode==='clean'?'on':'')+'" onclick="renderFinalCard(state.lastResult,\'clean\')">Clean</button>'+
      '<button type="button" class="'+(mode==='diff'?'on':'')+'" onclick="renderFinalCard(state.lastResult,\'diff\')">Show changes</button>'+
    '</div></div>'+
    '<div class="pvcard clean">'+body+'</div>'+
    '<div class="rp-actions">'+
    (changed?'<button class="btn small" onclick="applyAll()">&#10003; Apply all</button>':'')+
    '<button class="btn-copy" onclick="copyFinal()">Copy</button>'+
    '<button class="speakbtn" onclick="speak(\'corr\')">Speak</button></div>';
}
function applyAll(){
  const idx=[];
  state.errors.forEach(function(e,i){ if(!state.applied[i]) idx.push(i); });
  if(!idx.length){
    if(state.good){ document.getElementById('text').value=state.good; updateCounts(); pvRender(); }
    return;
  }
  idx.forEach(function(i){ accept(i); });
  if(state.lastResult) renderFinalCard(state.lastResult,'clean');
  pvRender();
}
function copyToClip(txt,btn){
  if(!txt) return;
  var mark=function(){
    if(!btn) return;
    var old=btn.innerHTML;
    btn.innerHTML='&#10003; Copied!';btn.classList.add('copied');
    setTimeout(function(){btn.innerHTML=old;btn.classList.remove('copied');},1800);
  };
  var fallback=function(){
    var ta=document.createElement('textarea');
    ta.value=txt;ta.style.position='fixed';ta.style.opacity='0';
    document.body.appendChild(ta);ta.select();
    try{document.execCommand('copy');}catch(e){}
    document.body.removeChild(ta);mark();
  };
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(txt).then(mark,fallback);
  }else{fallback();}
}
function copyFinal(){
  copyToClip(state.good||'',document.querySelector('#finalCard .btn-copy'));
}
function rephrase(){
  const ta=document.getElementById('text'), t=ta.value.trim();
  const btn=document.getElementById('btnRephrase'), spin=document.getElementById('spin');
  if(!t){ ta.focus(); return; }
  btn.disabled=true; spin.style.display='inline-block';
  var pane=document.getElementById('rephrasePane');
  if(pane) pane.innerHTML='<div class="empty-assist"><span class="big">Rephrasing\u2026</span>Reading your text once more, then pouring a clearer version.</div>';
  fetch('/api/rephrase', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text: t }) })
    .then(function(r){ return r.json(); })
    .then(function(j){ renderRephrase(j); })
    .catch(function(){ if(pane) pane.innerHTML='<div class="empty-assist"><span class="big">Rephrase interrupted</span>Please try again.</div>'; })
    .finally(function(){ btn.disabled=false; spin.style.display='none'; });
}
function renderRephrase(j){
  var pane=document.getElementById('rephrasePane');
  if(!pane) return;
  if(!j || !j.success || !j.rephrased){ pane.innerHTML='<div class="empty-assist"><span class="big">Nothing to rephrase</span>'+(esc(j&&j.message||'Try again.')+'</div>'); return; }
  state.rephrased=j.rephrased;
  var same=(j.rephrased===j.original_text);
  pane.innerHTML='<div class="pvhead"><h3>&#10024; Rephrased text</h3><small>'+(j.ai_used?'AI rewrote it':'local fixes applied')+' &middot; '+
    (same?'unchanged':'improved')+'</small></div>'+
    '<div class="rp-grid">'+
      '<div class="rp-col orig"><span class="rp-tag">Original</span>'+esc(j.original_text)+'</div>'+
      '<div class="rp-col rep"><span class="rp-tag">Rephrased</span>'+esc(j.rephrased)+'</div>'+
    '</div>'+
    (j.notes?'<div class="rp-note">'+esc(j.notes)+'</div>':'')+
    '<div class="rp-actions">'+
    (same?'':'<button class="btn small" onclick="useRephrased()">&#10003; Use this version</button>')+
    '<button class="btn-copy" onclick="copyRephrased()">Copy</button>'+
    '<button class="speakbtn" onclick="speak(\'rep\')">Speak rephrased</button>'+
    '</div>';
  pane.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function useRephrased(){
  var ta=document.getElementById('text');
  if(state.rephrased){ ta.value=state.rephrased; updateCounts(); pvRender(); }
}
function copyRephrased(){
  copyToClip(state.rephrased||'',document.querySelector('#rephrasePane .btn-copy'));
}
function pairHtml(e, i){
  const tip = esc(e.message || e.explanation || (e.original + ' \u2192 ' + e.correct));
  return '<button type="button" class="pr" id="pr'+i+'"'+(state.applied[i]?' disabled':'')+
    ' onclick="accept('+i+')" title="'+tip+'">'+
    '<del>'+esc(e.original)+'</del><span class="prarr">&rarr;</span><ins>'+esc(e.correct)+'</ins></button>';
}
function issueHtml(e, i){
  const tag = e.consensus ? ' &middot; '+esc(e.consensus.toLowerCase()) : '';
  return '<div class="issue" id="iss'+i+'">' +
    '<div class="ibody">' +
      '<div class="ih"><span class="iconly ig-'+igOf(e.type)+'"></span>'+
        esc(e.type||'grammar')+ (e.rule_id ? ' &middot; '+esc(e.rule_id) : '') +'</div>' +
      '<div class="itxt"><span class="bad">'+esc(e.wrong)+'</span>' +
        '<span class="arr">&rarr;</span><span class="gud">'+esc(e.correct)+'</span></div>' +
      '<div class="why">'+esc(e.message||e.explanation||'')+'</div></div>' +
    '<button class="btn small '+(state.applied[i]?'accepted':'ghost')+'" id="acc'+i+'" '+
      (state.applied[i]?'disabled':'')+' onclick="accept('+i+')">'+
      '&#10003; '+(state.applied[i]?'Applied':'Accept')+'</button></div>';
}
function accept(i){
  const e = state.errors[i];
  if(!e || state.applied[i]) return;
  const ta = document.getElementById('text');
  let v = ta.value;
  const cut = v.slice(0, e.start) + e.correct + v.slice(e.end);
  state.applied[i] = true;
  ta.value = cut;
  updateCounts();
  const delta = e.correct.length - (e.original ? e.original.length : (e.end - e.start));
  for(let j=0;j<state.errors.length;j++){
    if(j!==i && !state.applied[j] && state.errors[j].start >= e.end){
      state.errors[j].start += delta; state.errors[j].end += delta;
    }
  }
  const btn = document.getElementById('acc'+i);
  if(btn){ btn.disabled = true; btn.classList.remove('ghost'); btn.classList.add('accepted'); btn.textContent = '\u2713 Applied'; }
  applyMark(i, true);
  pvRender();
}
function applyMark(i, on){
  const iss=document.getElementById('iss'+i);
  if(iss) iss.classList.toggle('done', !!on);
  const pr=document.getElementById('pr'+i);
  if(pr){ pr.classList.toggle('fixed', !!on); pr.disabled=!!on; }
}
function setScore(v){
  const num = document.getElementById('scoreNum');
  if(v == null){ num.textContent = '\u2013'; return; }
  num.textContent = v;
}
document.getElementById('text').addEventListener('input', function(){ updateCounts(); pvRender(); });
document.getElementById('btnReset').addEventListener('click', function(){
  const ta = document.getElementById('text');
  if(state.good && state.good !== ta.value && confirm('Accept the corrected text?')){
    ta.value = state.good; updateCounts(); return;
  }
  if(state.original){ ta.value = state.original; updateCounts(); }
  state.errors = []; state.applied = {}; state.cards = null; state.rephrased = '';
  state.typeCards = null; state.metaHtml = '';
  document.getElementById('assistBody').innerHTML =
    '<div class="empty-assist"><span class="big">Ready when you are</span>' +
    'Click <b>Check grammar</b> to review your text.</div>';
  document.getElementById('btnReset').style.display = 'none';
  document.getElementById('btnRephrase').style.display = 'none';
  var at=document.getElementById('typeTabs');
  if(at) at.style.display='none';
  var fcR=document.getElementById('finalCard');
  if(fcR) fcR.style.display='none';
  setScore(null);
  var ec2=document.getElementById('editorCorrected');
  if(ec2) ec2.innerHTML='';
  var pv2=document.getElementById('previewPane');
  if(pv2) pv2.innerHTML='';
  var rp3=document.getElementById('rephrasePane');
  if(rp3) rp3.innerHTML='';
  var wp=document.getElementById('wpop');
  if(wp) wp.style.display='none';
  stopSpeech();
});
function summaryBarHtml(errors){
  if(!errors||!errors.length) return '';
  var counts={};
  errors.forEach(function(e){ var g=grpOf(e.type); counts[g]=(counts[g]||0)+1; });
  var igc={Grammar:'grammar',Spelling:'spelling',Punctuation:'punctuation',
    Style:'style',Clarity:'clarity',Context:'context',Other:'other'};
  var h='<div class="summary-bar">';
  Object.keys(counts).forEach(function(g){
    h+='<span class="summary-chip"><span class="sdot ig-'+(igc[g]||'other')+'"></span>'+g+' '+counts[g]+'</span>';
  });
  return h+'</div>';
}
function correctedTextHtml(j){
  if(!j.corrected_text||j.corrected_text===j.original_text) return '';
  var ow=wordGuess(j.original_text),cw=wordGuess(j.corrected_text);
  var d=cw-ow,ds=d===0?'':d>0?' (+'+d+')':' ('+d+')';
  var dh=diffHtml(j);
  return '<div class="corrected-section" id="correctedSection">'+
    '<div class="corrected-label">&#10003; Corrected Text <small style="font-weight:400;text-transform:none;letter-spacing:0;color:#8a6a71">(changes shown inline)</small></div>'+
    '<div class="corrected-box diff" id="correctedBox">'+(dh ? dh : esc(j.corrected_text))+'</div>'+
    '<div class="corrected-meta"><b>'+cw+'</b> words'+ds+' &middot; <b>'+j.corrected_text.length+'</b> chars</div>'+
    '<div class="corrected-actions">'+
    '<button class="btn small" id="btnAcceptAll" onclick="acceptAll()">&#10003; Accept All</button>'+
    '<button class="btn-copy" id="btnCopy" onclick="copyCorrected()">Copy</button>'+
    '</div></div>';
}
function diffHtml(j){
  var errs=(j.errors||[]).slice().sort(function(a,b){ return (a.start-b.start)||(a.end-b.end); });
  if(!errs.length) return '';
  var h='', pos=0, any=false;
  errs.forEach(function(e){
    if(e.start<pos||e.start>=e.end) return;
    h+=esc(j.original_text.slice(pos,e.start));
    h+='<span class="del">'+esc(j.original_text.slice(e.start,e.end))+'</span>'+
       '<span class="ins">'+esc(e.correct)+'</span>';
    pos=e.end; any=true;
  });
  if(!any) return '';
  return h+esc(j.original_text.slice(pos));
}
function acceptAll(){
  var ta=document.getElementById('text');
  if(state.good){ta.value=state.good;updateCounts();}
  for(var i=0;i<state.errors.length;i++){
    state.applied[i]=true;
    var btn=document.getElementById('acc'+i);
    if(btn){btn.disabled=true;btn.classList.remove('ghost');btn.classList.add('accepted');btn.textContent='\u2713 Applied';}
    var iss=document.getElementById('iss'+i);
    if(iss) iss.classList.add('done');
  }
  var ab=document.getElementById('btnAcceptAll');
  if(ab){ab.disabled=true;ab.textContent='\u2713 All Applied';}
  pvRender();
}
function copyCorrected(){
  copyToClip(state.good||document.getElementById('text').value,
             document.getElementById('btnCopy'));
}
updateCounts();

/* ---------------- In-line review mirror (click an error to fix it) ---------------- */
let tipEl=null, tipFor=-1;
function pvRender(){
  const pv=document.getElementById('previewPane');
  if(!pv) return;
  const ta=document.getElementById('text');
  const txt=ta.value;
  if(!txt || !state.errors || !state.errors.length){ pv.innerHTML=''; return; }
  const live=[];
  state.errors.forEach(function(e,i){ if(!state.applied[i] && e && e.start<e.end && e.start>=0) live.push({e:e,i:i}); });
  if(!live.length){ pv.innerHTML=''; return; }
  live.sort(function(a,b){ return a.e.start-b.e.start; });
  let h='<div class="pvhead"><h3>&#8226; In-line review &mdash; click an error to fix it</h3><small>'+live.length+' remaining</small></div><div class="pvcard">';
  let pos=0;
  live.forEach(function(it){
    const e=it.e, i=it.i;
    if(e.start<pos) return;
    h+=esc(txt.slice(pos,e.start));
    const seg=esc(txt.slice(e.start,e.end)) || esc(e.wrong||'\uFFFD');
    h+='<mark class="e '+igOf(e.type)+'" data-i="'+i+'">'+seg+'</mark>';
    pos=e.end;
  });
  h+=esc(txt.slice(pos));
  h+='</div>';
  pv.innerHTML=h;
  if(!tipEl){
    tipEl=document.createElement('div');
    tipEl.className='tip';
    tipEl.addEventListener('click',function(ev){
      const b=ev.target;
      if(!tipEl.contains(b)) return;
      if(b.className==='tclose'){ hideTip(); return; }
      if(b.className==='tfix'){
        const i=parseInt(b.getAttribute('data-i'),10);
        hideTip();
        accept(i);
      }
    });
    document.body.appendChild(tipEl);
  }
  pv.querySelectorAll('.e').forEach(function(m){
    m.addEventListener('mouseenter',function(){ showTip(m); });
    m.addEventListener('click',function(){ showTip(m); });
    m.addEventListener('mouseleave',function(){ setTimeout(function(){ if(tipFor!==m.getAttribute('data-i')) hideTip(); },400); });
  });
}
function showTip(m){
  const i=parseInt(m.getAttribute('data-i'),10), e=state.errors[i];
  if(!e) return;
  tipFor=i;
  tipEl.innerHTML='<button class="tclose">&times;</button><b>'+esc(e.type||'grammar')+'</b>'+
    (e.rule_id? esc(' &middot; '+e.rule_id):'')+
    '<div style="margin:4px 0">'+esc(e.wrong)+' &rarr; <b style="color:var(--good)">'+esc(e.correct)+'</b></div>'+
    '<div>'+(esc(e.message||e.explanation||''))+'</div>'+
    '<button class="tfix" data-i="'+i+'">&#10003; Fix here</button>';
  const r=m.getBoundingClientRect();
  tipEl.style.display='block';
  tipEl.style.top=Math.round(r.bottom+6+window.scrollY)+'px';
  tipEl.style.left=Math.max(8,Math.min(r.left+window.scrollX, window.innerWidth-tipEl.offsetWidth-10))+'px';
}
function hideTip(){ if(tipEl){ tipEl.style.display='none'; tipFor=-1; } }

/* ---------------- Text to speech (read aloud) ---------------- */
let voices=[];
function loadVoices(){
  if(!('speechSynthesis' in window)) return;
  voices=speechSynthesis.getVoices();
  const sel=document.getElementById('voiceSel');
  if(!sel) return;
  const cur=sel.value;
  const en=voices.filter(function(v){ return /^en/i.test(v.lang); });
  const list=en.length?en:voices;
  if(!list.length) return;
  sel.innerHTML=list.map(function(v){
    return '<option value="'+esc(v.name)+'">'+esc(v.name)+'</option>';
  }).join('');
  if(cur){ sel.value=cur; }
  else{
    let v=list[0];
    for(let i=0;i<list.length;i++){ if(/en[-_]US/i.test(list[i].lang)){ v=list[i]; break; } }
    sel.value=v.name;
  }
}
function chosenVoice(){
  const sel=document.getElementById('voiceSel');
  const n=sel?sel.value:'';
  if(!n) return null;
  for(let i=0;i<voices.length;i++){ if(voices[i].name===n) return voices[i]; }
  return null;
}
function speak(which){
  if(!('speechSynthesis' in window)) return;
  stopSpeech();
  const ta=document.getElementById('text');
  const j=state.lastResult;
  const txt=(which==='corr'&&j&&j.corrected_text)? j.corrected_text
         : (which==='orig'&&j)? j.original_text
         : (which==='rep')? (state.rephrased||ta.value) : ta.value;
  if(!txt || !txt.trim()) return;
  const u=new SpeechSynthesisUtterance(txt);
  const v=chosenVoice();
  if(v){ u.voice=v; u.lang=v.lang; } else { u.lang='en-US'; }
  u.rate=0.98; u.pitch=1;
  speechSynthesis.speak(u);
  document.getElementById('btnStop').style.display='inline-block';
  const btn=document.getElementById(which==='corr'?'btnSpc':(which==='rep'?'btnSpo':'btnSpo'));
  if(btn) btn.classList.add('stop');
  u.onend=u.onerror=function(){ stopSpeech(); };
}
function stopSpeech(){
  if('speechSynthesis' in window) speechSynthesis.cancel();
  var b=document.getElementById('btnStop');
  if(b) b.style.display='none';
  var b1=document.getElementById('btnSpc'), b2=document.getElementById('btnSpo');
  if(b1) b1.classList.remove('stop');
  if(b2) b2.classList.remove('stop');
}

/* ---------------- Speech to text (dictation) ---------------- */
let rec=null, recOn=false;
(function(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const btn=document.getElementById('micBtn');
  if(!SR || !btn){ if(btn) btn.style.display='none'; return; }
  btn.addEventListener('click', function(){
    if(recOn){ if(rec) rec.stop(); return; }
    const ta=document.getElementById('text');
    rec=new SR();
    rec.lang='en-US'; rec.interimResults=true; rec.continuous=true;
    rec.onstart=function(){ recOn=true; btn.classList.add('rec'); btn.textContent='Listening\u2026'; };
    rec.onend=function(){ recOn=false; btn.classList.remove('rec'); btn.textContent='Dictate'; };
    rec.onerror=function(){ recOn=false; btn.classList.remove('rec'); btn.textContent='Dictate'; };
    rec.onresult=function(ev){
      let text='';
      for(let i=0;i<ev.results.length;i++){ text+=ev.results[i][0].transcript; }
      ta.value=text;
      updateCounts(); pvRender();
    };
    rec.start();
  });
})();

/* ---------------- Word tools: synonyms & antonyms (Datamuse) ---------------- */
let wpopWord='', wpopStart=-1, wpopEnd=-1;
function hideWpop(){
  const p=document.getElementById('wpop');
  if(p) p.style.display='none';
}
function openWordTools(pos){
  const ta=document.getElementById('text'), t=ta.value;
  if(!t || pos<0) return;
  const re=/[A-Za-z][A-Za-z']*/g;
  let mm=null, hit=null;
  while((mm=re.exec(t))!==null){
    if(pos<=re.lastIndex){ hit={ w:t.slice(mm.index,re.lastIndex), s:mm.index, e:re.lastIndex }; break; }
  }
  if(!hit) return;
  wpopWord=hit.w; wpopStart=hit.s; wpopEnd=hit.e;
  document.getElementById('wpopWord').textContent='“'+hit.w+'”';
  wtab('syn');
  const r=ta.getBoundingClientRect();
  const p=document.getElementById('wpop');
  p.style.display='block';
  p.style.left=Math.max(8,Math.min(r.left+window.scrollX, window.innerWidth-p.offsetWidth-10))+'px';
  p.style.top=Math.round(r.bottom+8+window.scrollY)+'px';
}
function wtab(tab){
  document.getElementById('tabSyn').classList.toggle('on', tab==='syn');
  document.getElementById('tabAnt').classList.toggle('on', tab==='ant');
  const box=document.getElementById('wchips');
  if(!wpopWord){ box.innerHTML='<span style="cursor:default;opacity:.6">Double-click a word above to search it.</span>'; return; }
  box.innerHTML='<span style="cursor:default;opacity:.6">loading\u2026</span>';
  fetch('https://api.datamuse.com/words?'+(tab==='syn'?'rel_syn':'rel_ant')+'='+encodeURIComponent(wpopWord)+'&max=16&md=p')
    .then(function(r){ return r.json(); })
    .then(function(list){
      let items=list.slice(0,15);
      if(tab==='syn') items.sort(function(a,b){
        const ka=(a.tags&&a.tags.indexOf('common')>-1)?1:0;
        const kb=(b.tags&&b.tags.indexOf('common')>-1)?1:0;
        return kb-ka;
      });
      if(!items.length){ box.innerHTML='<span style="cursor:default;opacity:.6">no '+(tab==='syn'?'synonyms':'antonyms')+' found</span>'; return; }
      box.innerHTML=items.map(function(x){
        return '<span title="'+esc((x.defs&&x.defs[0]||'').replace(/^[^:]*:/,''))+'">'+esc(x.word)+'</span>';
      }).join('');
      box.querySelectorAll('span[title]').forEach(function(s){
        s.addEventListener('click', function(){
          const ta=document.getElementById('text');
          const v=ta.value;
          let rep=s.textContent;
          const orig=v.slice(wpopStart,wpopEnd);
          if(orig && /^[A-Z]/.test(orig)) rep=rep.charAt(0).toUpperCase()+rep.slice(1);
          const delta=rep.length-(wpopEnd-wpopStart);
          ta.value=v.slice(0,wpopStart)+rep+v.slice(wpopEnd);
          if(state.errors){ state.errors.forEach(function(e){ if(e.start>=wpopEnd){ e.start+=delta; e.end+=delta; } }); }
          updateCounts(); pvRender(); hideWpop();
        });
      });
    })
    .catch(function(){ box.innerHTML='<span style="cursor:default;opacity:.6">could not reach Datamuse &#8212; check your connection.</span>'; });
}
document.getElementById('text').addEventListener('dblclick', function(ev){
  const ta=document.getElementById('text');
  openWordTools(ta.selectionStart!==ta.selectionEnd? ta.selectionEnd : ta.selectionEnd);
});
document.addEventListener('mousedown', function(ev){
  const p=document.getElementById('wpop');
  if(p && p.style.display==='block' && !p.contains(ev.target) && ev.target.id!=='text') hideWpop();
});
document.addEventListener('keydown', function(ev){
  if(ev.key==='Escape'){ hideWpop(); hideTip(); }
});

/* ---------------- init ---------------- */
if('speechSynthesis' in window){
  const sb=document.getElementById('speechBar');
  if(sb) sb.style.display='flex';
  loadVoices();
  if(speechSynthesis.onvoiceschanged!==undefined) speechSynthesis.onvoiceschanged=loadVoices;
}else{
  const sb=document.getElementById('speechBar');
  if(sb) sb.style.display='none';
}
if(document.getElementById('voiceSel')) document.getElementById('voiceSel').addEventListener('change', stopSpeech);
document.addEventListener('keydown',function(e){if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();check();}});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def root():
    return Response(_INDEX_HTML, status=200, mimetype="text/html")


def _ai_budget() -> float:
    try:
        return max(2.0, float(os.environ.get("GC_AI_BUDGET", "") or "8"))
    except (TypeError, ValueError):
        return 8.0


def _run_with_budget(fn, budget: float):
    """Run ``fn`` in a worker thread and wait at most ``budget`` seconds.

    Returns fn's result when it finishes in time, else None (the caller then
    returns its already-computed fast baseline). Keeps /api/check snappy even
    when the AI provider is slow or quota-limited.
    """
    box = {}
    done = threading.Event()

    def worker():
        try:
            box["value"] = fn()
        except Exception:  # pragma: no cover - defensive
            box["error"] = True
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    if done.wait(budget):
        return box.get("value")
    return None


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
        max_passes = int(os.environ.get("GC_MAX_PASSES", "") or "1")
        ai_wanted = payload.get("use_ai", True) and ai_key_configured()
        if ai_wanted:
            # Fast baseline first: the deterministic rule path returns in a
            # fraction of a second and already covers common learner errors.
            offline = check_ai_text(text, use_ai=False)
            # Upgrade to the full AI pass only if it finishes within the
            # budget — otherwise reply with the fast offline result instead of
            # making the user wait on a slow/unstable AI provider.
            ai = _run_with_budget(
                lambda: check_ai_text(text, use_ai=True, max_passes=max_passes),
                _ai_budget())
            result = ai if (ai is not None and ai.get("success")) else offline
            if result is not offline:
                result.setdefault("meta", {})["ai_racing"] = False
        else:
            result = check_ai_text(text, use_ai=False)
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


_REPHRASE_PROMPT = (
    "You are a professional English rewriting assistant for non-native learners. "
    "Rewrite the text below into clear, natural, fluent English.\n"
    "HARD RULES:\n"
    "1. Keep the writer's EXACT meaning and every idea, in the same order.\n"
    "2. Fix all grammar, spelling and punctuation errors.\n"
    "3. Prefer simple, confident wording. Do not add new information or opinions.\n"
    "4. Keep roughly the same length as the original.\n"
    "5. Do NOT change names, numbers, places or quoted speech.\n"
    "Return ONLY strict JSON: "
    '{"original_text": "", "rephrased": "", "notes": ""}\n'
    "Text to rewrite:\n"
)


def _rephrase_text(text: str) -> dict:
    started = time.time()
    if not text or not text.strip():
        return {"success": False, "message": "empty text", "original_text": text,
                "rephrased": "", "ai_used": False,
                "processing_time_ms": int((time.time() - started) * 1000)}
    call = _provider_callable()
    if call is not None:
        raw = _run_with_budget(lambda: call(_REPHRASE_PROMPT + text),
                               _ai_budget())
        if raw:
            data = _extract_json(raw)
            if data and isinstance(data, dict):
                rephrased = str(data.get("rephrased") or "").strip()
                if rephrased:
                    return {"success": True, "original_text": text, "rephrased": rephrased,
                            "ai_used": True, "notes": str(data.get("notes") or ""),
                            "processing_time_ms": int((time.time() - started) * 1000)}
    # Fallback: apply the local grammar fixes as a light rewrite when the AI
    # model is unavailable or fails (never worse than the original).
    try:
        fix = check_ai_text(text, use_ai=False, max_passes=1)
        rephrased = fix.get("corrected_text") or text
    except Exception:
        rephrased = text
    return {"success": True, "original_text": text, "rephrased": rephrased,
            "ai_used": False,
            "processing_time_ms": int((time.time() - started) * 1000)}


@app.route("/api/rephrase", methods=["POST"])
def api_rephrase():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "") or "").strip()
    if not text:
        return Response(
            json.dumps({"success": False, "message": "empty text",
                        "original_text": "", "rephrased": ""},
                       ensure_ascii=False),
            status=400, mimetype="application/json")
    return Response(json.dumps(_rephrase_text(text), ensure_ascii=False),
                    status=200, mimetype="application/json")


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
                body = ""
                try:
                    body = exc.read().decode()[:1500]
                except Exception:
                    pass
                if body:
                    info["call_response_body"] = body
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