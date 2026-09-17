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
<!DOCTYPE html>
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
  --spel:#c2456b; --gram:#7a2631; --punc:#8a3a4b; --styl:#9c7a2e; --ctx:#9c6b77;
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
.topbar { position:relative; z-index:2; display:flex; align-items:center; gap:14px;
  padding:12px 22px; background:var(--panel); backdrop-filter:blur(14px);
  border-bottom:1px solid rgba(114,47,55,.18); }
.topbar .crest-s { width:36px; height:36px; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, var(--wine-light), var(--wine-dark));
  display:flex; align-items:center; justify-content:center; flex:none;
  box-shadow:0 6px 16px rgba(114,47,55,.35), inset 0 0 0 1.5px rgba(255,255,255,.3); }
.topbar .crest-s svg { width:18px; height:18px; }
.brand { font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.5rem;
  background:linear-gradient(120deg, var(--wine-dark), var(--wine) 45%, var(--wine-light));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.brand small { display:block; font-family:'Outfit',sans-serif; font-size:.66rem;
  font-weight:500; letter-spacing:.16em; text-transform:uppercase; color:var(--wine); }
.topright { margin-left:auto; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.ai-toggle { display:inline-flex; align-items:center; gap:7px; font-size:.8rem; color:var(--wine);
  border:1px solid rgba(114,47,55,.3); border-radius:99px; padding:6px 13px; cursor:pointer;
  background:rgba(255,255,255,.5); user-select:none; }
.ai-toggle input { accent-color:var(--wine); }
.scorepill { display:inline-flex; align-items:center; gap:7px; font-weight:700; font-size:.84rem;
  border-radius:99px; padding:6px 14px; color:var(--cream); border:1px solid rgba(255,255,255,.25);
  background:linear-gradient(135deg, var(--wine-light), var(--wine) 55%, var(--wine-dark));
  box-shadow:0 6px 16px rgba(114,47,55,.3); min-width:96px; justify-content:center; }
.scorepill .sc-num { font-size:1.05rem; }
.wrap { position:relative; z-index:1; max-width:1180px; margin:0 auto; padding:28px 20px 70px; }
.grid { display:grid; grid-template-columns:minmax(0,9fr) minmax(0,6fr); gap:24px;
  align-items:start; }
@media (max-width:900px){ .grid { grid-template-columns:1fr; } }
.panel { position:relative; padding:22px; border-radius:22px; background:var(--panel);
  backdrop-filter:blur(14px); border:1px solid rgba(114,47,55,.18);
  box-shadow:0 20px 50px rgba(74,26,32,.14), inset 0 1px 0 rgba(255,255,255,.85);
  animation:rise .6s ease both; }
.panel::before { content:""; position:absolute; inset:0; border-radius:22px; padding:1px;
  background:linear-gradient(135deg, rgba(201,165,100,.7), transparent 30%, transparent 70%, rgba(114,47,55,.55));
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none; }
.editor { padding:18px; }
label { display:block; font-size:.74rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--wine); font-weight:600; margin-bottom:10px; }
.edbox { position:relative; }
#text { width:100%; min-height:190px; resize:vertical; border-radius:14px;
  border:1px solid rgba(114,47,55,.26); background:rgba(255,255,255,.78);
  padding:15px 16px; font-size:1.05rem; color:var(--ink); line-height:1.7;
  font-family:'Outfit',sans-serif; outline:none; position:relative; z-index:2; }
#text:focus { border-color:var(--wine); box-shadow:0 0 0 4px rgba(169,90,99,.16); }
.edbar { display:flex; align-items:center; gap:10px; margin-top:12px; flex-wrap:wrap; }
.counts { font-size:.8rem; color:#8a6a71; margin-right:auto; }
.counts b { color:var(--wine); }
.samples { display:flex; gap:8px; flex-wrap:wrap; }
.samples button { background:transparent; color:var(--wine); border:1px solid rgba(114,47,55,.32);
  border-radius:99px; padding:6px 13px; font-size:.78rem; cursor:pointer;
  font-family:'Outfit',sans-serif; transition:all .25s; }
.samples button:hover { background:var(--wine); color:var(--cream); transform:translateY(-2px);
  box-shadow:0 8px 18px rgba(114,47,55,.3); }
.btn { position:relative; overflow:hidden; border:none; cursor:pointer; border-radius:14px;
  padding:11px 24px; font-size:.98rem; font-weight:600; font-family:'Outfit',sans-serif;
  color:var(--cream);
  background:linear-gradient(135deg, var(--wine-light), var(--wine) 55%, var(--wine-dark));
  box-shadow:0 10px 24px rgba(114,47,55,.34); transition:transform .2s, box-shadow .25s; }
.btn::after { content:""; position:absolute; top:0; left:-80%; width:50%; height:100%;
  background:linear-gradient(120deg, transparent, rgba(255,255,255,.38), transparent);
  transform:skewX(-20deg); }
.btn:hover { transform:translateY(-2px); box-shadow:0 14px 32px rgba(114,47,55,.42); }
.btn:hover::after { animation:sheen 1s ease; }
.btn:disabled { opacity:.6; cursor:progress; transform:none; box-shadow:none; }
.btn.ghost { background:transparent; color:var(--wine); border:1px solid rgba(114,47,55,.3);
  box-shadow:none; }
.btn.ghost:hover { background:rgba(114,47,55,.08); }
.btn.small { padding:8px 14px; font-size:.86rem; border-radius:12px; }
.btn.small.accepted { background:rgba(46,125,91,.85); box-shadow:none; cursor:default; }
.btn.small.accepted:hover { transform:none; }
.spin { display:none; width:18px; height:18px; border:3px solid rgba(114,47,55,.2);
  border-top-color:var(--wine); border-radius:50%; animation:rot .7s linear infinite; }

.assistant { position:sticky; top:20px; padding:20px; max-height:calc(100vh - 100px);
  display:flex; flex-direction:column; }
.assist-head { display:flex; align-items:center; gap:6px; }
.assist-head h2 { margin:0; font-size:.8rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--wine); font-weight:700; display:flex; align-items:center; gap:9px; flex:1; }
.assist-head h2 .dot { width:9px; height:9px; border-radius:50%; background:var(--gold); flex:none;
  box-shadow:0 0 0 4px rgba(201,165,100,.24); }
.status { font-size:.82rem; color:#8a6a71; display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
.status b { color:var(--wine); }
.chips { margin:10px 0 4px; display:flex; gap:6px; flex-wrap:wrap; justify-content:center; }
.chip { background:rgba(114,47,55,.1); border-radius:8px; padding:3px 9px; font-size:.72rem;
  letter-spacing:.04em; font-weight:600; color:var(--wine);
  border:1px solid rgba(114,47,55,.2); }
.assist-scroll { overflow-y:auto; margin-top:6px; padding-right:4px; }
.group { margin-top:14px; }
.grouptitle { display:flex; align-items:center; gap:8px; font-size:.78rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.08em; color:var(--wine-dark); margin-bottom:8px; }
.grouptitle .gbadge { font-size:.7rem; border-radius:99px; padding:1px 8px; color:var(--cream);
  background:var(--wine); }
.gempty { font-size:.84rem; color:#9a7a82; padding:8px 2px 14px; }
.iconly { width:10px; height:10px; border-radius:3px; flex:none; }
.ig-spelling { background:var(--spel); } .ig-grammar { background:var(--gram); }
.ig-punctuation { background:var(--punc); } .ig-style { background:var(--styl); }
.ig-context,.ig-other { background:var(--ctx); }
.issue { display:flex; gap:10px; align-items:flex-start; padding:11px 12px; border-radius:12px;
  background:rgba(255,255,255,.6); border:1px solid rgba(114,47,55,.12);
  margin-bottom:9px; transition:all .25s; animation:rise .4s ease both; }
.issue.done { opacity:.62; background:rgba(46,125,91,.08); border-color:rgba(46,125,91,.25); }
.issue .ibody { flex:1; min-width:0; }
.issue .ih { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:600;
  color:var(--wine-light); display:flex; align-items:center; gap:6px; }
.issue .itxt { font-size:.96rem; font-weight:600; margin-top:5px; }
.issue .itxt .bad { color:var(--bad); text-decoration:line-through;
  text-decoration-color:rgba(181,68,110,.6); }
.issue .itxt .arr { color:var(--gold); margin:0 5px; }
.issue .itxt .gud { color:var(--good); }
.issue .why { font-size:.8rem; color:#6b4a52; margin-top:5px; line-height:1.4; }
.issue .chip { background:rgba(114,47,55,.07); font-weight:500; font-size:.68rem; }
.wrap-dark { color:var(--ink); margin-top:40px; text-align:center; }
.footer { text-align:center; color:#8a6a71; font-size:.9rem; margin-top:40px;
  font-family:'Cormorant Garamond',serif; font-style:italic; animation:rise .8s .4s ease both; }
.empty-assist { color:#9a7a82; font-size:.88rem; text-align:center; padding:26px 6px; line-height:1.6; }
.empty-assist .big { font-family:'Cormorant Garamond',serif; font-size:1.5rem; font-style:italic;
  color:var(--wine); display:block; margin-bottom:6px; }
@keyframes rise { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:none; } }
@keyframes rot { to { transform:rotate(360deg); } }
@keyframes sheen { to { left:130%; } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }
.summary-bar{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 4px;justify-content:center}
.summary-chip{display:inline-flex;align-items:center;gap:5px;background:rgba(114,47,55,.07);
  border:1px solid rgba(114,47,55,.14);border-radius:8px;padding:4px 10px;font-size:.72rem;
  font-weight:600;color:var(--wine)}
.summary-chip .sdot{width:8px;height:8px;border-radius:50%;flex:none}
.corrected-section{margin-top:16px;padding-top:16px;border-top:1px dashed rgba(114,47,55,.18);
  animation:rise .4s ease both}
.corrected-label{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--good);
  font-weight:700;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.corrected-box{background:rgba(46,125,91,.06);border:1px solid rgba(46,125,91,.22);
  border-radius:14px;padding:14px 16px;font-size:1rem;line-height:1.7;color:var(--ink);
  min-height:60px;white-space:pre-wrap;word-break:break-word}
.corrected-meta{font-size:.74rem;color:#8a6a71;margin-top:8px}
.corrected-meta b{color:var(--good)}
.corrected-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.btn-copy{background:rgba(114,47,55,.08);color:var(--wine);border:1px solid rgba(114,47,55,.25);
  border-radius:12px;padding:8px 16px;font-size:.86rem;cursor:pointer;font-family:'Outfit',sans-serif;
  font-weight:500;transition:all .2s}
.btn-copy:hover{background:rgba(114,47,55,.15);transform:translateY(-1px)}
.btn-copy.copied{background:rgba(46,125,91,.12);color:var(--good);border-color:rgba(46,125,91,.3)}
.kbd-hint{display:inline-block;font-size:.65rem;background:rgba(114,47,55,.08);border:1px solid rgba(114,47,55,.15);
  border-radius:4px;padding:1px 5px;margin-left:6px;color:var(--wine-light);vertical-align:middle;opacity:.7}
.editor-corrected{margin-top:20px}
</style>
</head>
<body>
<div class="grain"></div>
<div class="topbar">
  <div class="crest-s">
    <svg viewBox="0 0 24 24" fill="none" stroke="#faf3f0" stroke-width="1.6">
      <path d="M12 3l2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 8.7l5.4-.8L12 3z"/>
    </svg>
  </div>
  <div class="brand">WriteMaster AI<small>refined grammar, poured slowly</small></div>
  <div class="topright">
    <label class="ai-toggle" title="Final validation by Google Gemini (needs a configured key)">
      <input type="checkbox" id="useAi"> AI validation
    </label>
    <div class="scorepill" title="Writing quality estimate"><span class="sc-num" id="scoreNum">&ndash;</span>/ 100</div>
  </div>
</div>

<div class="wrap">
  <div class="grid">
    <div class="panel editor">
      <label for="text">Your text</label>
      <div class="edbox">
        <textarea id="text" placeholder="Type or paste a sentence…  e.g.  She go to school everyday and she dont like it"></textarea>
      </div>
      <div class="edbar">
        <div class="counts"><b id="cChars">0</b> chars &middot; <b id="cWords">0</b> words &middot; <b id="cSents">0</b> sentences <span class="kbd-hint">Ctrl+Enter</span></div>
        <button class="btn ghost small" id="btnReset" style="display:none">Reset</button>
        <button class="btn" id="btn" onclick="check()">&#10005; Check grammar</button>
        <span class="spin" id="spin"></span>
      </div>
      <div class="samples" style="margin-top:14px">
        <button type="button" onclick="setSample(0)">She go to school…</button>
        <button type="button" onclick="setSample(1)">I have a good time yesterday.</button>
        <button type="button" onclick="setSample(2)">He dont like coffee</button>
        <button type="button" onclick="setSample(3)">The men is walking fast</button>
      </div>
      <div id="editorCorrected" class="editor-corrected"></div>
    </div>

    <div class="panel assistant">
      <div class="assist-head">
        <h2><span class="dot"></span>Assistant</h2>
      </div>
      <div id="assistBody"></div>
    </div>
  </div>
  <div class="footer">&#8226; verified by Google Gemini &#8226; every word mellowed to perfection &#8226;</div>
</div>
<script>
const samples = [
  "She go to school everyday and she dont like it",
  "I have a good time yesterday.",
  "He dont like coffee",
  "The men is walking fast"
];
let state = { errors: [], applied: {} };
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
    return;
  }
  const groups = {};
  errors.forEach(function(e, i){ ++(groups[grpOf(e.type)] = groups[grpOf(e.type)] || {}).n ||
    (groups[grpOf(e.type)] = { n:1, list:[] }).list; groups[grpOf(e.type)].list.push({e:e, i:i}); });
  let html = meta;
  Object.keys(groups).forEach(function(g){
    const gd = groups[g];
    html += '<div class="group"><div class="grouptitle"><span class="iconly ig-'+igOf(gd.list[0].e.type)+'"></span>' +
      g + ' <span class="gbadge">'+gd.n+'</span></div>';
    gd.list.forEach(function(item){ html += issueHtml(item.e, item.i); });
    html += '</div>';
  });
  body.innerHTML = html;
  for(const i in state.applied) applyMark(i, !!state.applied[i]);
  var ec=document.getElementById('editorCorrected');
  if(ec) ec.innerHTML = correctedTextHtml(j);
  var cs=document.getElementById('correctedSection');
  if(cs) setTimeout(function(){cs.scrollIntoView({behavior:'smooth',block:'nearest'});},150);
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
  btn.disabled = true; btn.classList.remove('ghost'); btn.classList.add('accepted');
  btn.textContent = '\u2713 Applied';
  document.getElementById('iss'+i).classList.add('done');
}
function setScore(v){
  const num = document.getElementById('scoreNum');
  if(v == null){ num.textContent = '\u2013'; return; }
  num.textContent = v;
}
document.getElementById('text').addEventListener('input', updateCounts);
document.getElementById('btnReset').addEventListener('click', function(){
  const ta = document.getElementById('text');
  if(state.good && state.good !== ta.value && confirm('Accept the corrected text?')){
    ta.value = state.good; updateCounts(); return;
  }
  if(state.original){ ta.value = state.original; updateCounts(); }
  state.errors = []; state.applied = {};
  document.getElementById('assistBody').innerHTML =
    '<div class="empty-assist"><span class="big">Ready when you are</span>' +
    'Click <b>Check grammar</b> to review your text.</div>';
  document.getElementById('btnReset').style.display = 'none';
  setScore(null);
  var ec2=document.getElementById('editorCorrected');
  if(ec2) ec2.innerHTML='';
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
  return '<div class="corrected-section" id="correctedSection">'+
    '<div class="corrected-label">&#10003; Corrected Text</div>'+
    '<div class="corrected-box" id="correctedBox">'+esc(j.corrected_text)+'</div>'+
    '<div class="corrected-meta"><b>'+cw+'</b> words'+ds+' &middot; <b>'+j.corrected_text.length+'</b> chars</div>'+
    '<div class="corrected-actions">'+
    '<button class="btn small" id="btnAcceptAll" onclick="acceptAll()">&#10003; Accept All</button>'+
    '<button class="btn-copy" id="btnCopy" onclick="copyCorrected()">Copy</button>'+
    '</div></div>';
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
}
function copyCorrected(){
  var txt=state.good||document.getElementById('text').value;
  var ta=document.createElement('textarea');
  ta.value=txt;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.select();
  try{document.execCommand('copy');showCopied();}catch(e){}
  document.body.removeChild(ta);
}
function showCopied(){
  var b=document.getElementById('btnCopy');
  b.textContent='Copied!';b.classList.add('copied');
  setTimeout(function(){b.textContent='Copy';b.classList.remove('copied');},2000);
}
updateCounts();
document.addEventListener('keydown',function(e){if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();check();}});
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
                               if ai_key_configured() else False,
                               max_passes=3)
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