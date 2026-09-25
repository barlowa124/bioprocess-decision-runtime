HTML = r'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bioprocess | Evidence Studio</title><link rel="stylesheet" href="/app.css"><script src="/app.js" defer></script></head>
<body>
<aside class="sidebar"><div class="brand"><span class="brand-mark">B</span><div>Bioprocess<span class="subbrand">Evidence Studio</span></div></div><div class="sidebar-label">DEMONSTRATION</div>
<nav aria-label="Demo views"><button id="nav-evidence" class="nav active" aria-current="page">Numerical evidence</button><button id="nav-advisory" class="nav">Synthetic advisory</button></nav>
<div class="sidebar-bottom"><span class="dot"></span> Local demonstration<div>No cloud inference<br>No equipment connections</div></div></aside>
<main>
<header><div><div class="eyebrow">INTERPRETABLE-BY-CONSTRUCTION</div><h1 id="page-title">Execution evidence</h1><p id="page-subtitle">Inspect what was computed, compared, and recorded.</p></div><span id="mode-badge" class="badge blue">SAVED EVIDENCE</span></header>
<div class="notice"><strong>Research prototype.</strong> Synthetic data and bounded numerical experiments only. Not validated for clinical, GMP, manufacturing, or process-control use.</div>
<div id="error" class="error" role="alert" hidden></div>
<section id="view-evidence" aria-labelledby="page-title">
<div class="metrics"><div class="metric"><span>Connected decoder coverage</span><strong id="connected-count">—</strong><small>Saved cases; continuation disclosed</small></div><div class="metric"><span>Current frontier</span><strong id="frontier">—</strong><small id="frontier-note">Coverage from pinned saved evidence</small></div><div class="metric"><span>Verification mode</span><strong>Saved records</strong><small>No fresh numerical inference in this UI</small></div></div>
<section class="card"><div class="section-heading"><div><h2>Decoder coverage</h2><p>Gemma 3 270M · 18 layers · indices are zero-based</p></div><span class="muted">Independent arithmetic evidence shown</span></div><div id="layers" class="layers" aria-label="Decoder layers"></div><div class="legend"><span><i class="swatch connected"></i>Connected case evidence</span><span><i class="swatch partial"></i>Partial, reused boundaries</span><span><i class="swatch pending"></i>No independent result shown</span></div><p id="layer-note" class="small">Full-attention layers are 5, 11, and 17. Coverage does not imply language-task or hardware qualification.</p></section>
<div class="evidence-layout"><section class="card experiments"><h2>Saved experiments</h2><p>Choose an evidence package.</p><div id="experiments"></div><div class="divider"></div><div class="small">Dashboard checks pinned compact artifact hashes and references only. It does not revalidate raw tensors, rerun the numerical verifier, or authenticate an external signature.</div></section><section id="evidence-detail" class="card" aria-live="polite"><p>Loading saved evidence…</p></section></div>
<section class="card verification"><div><h2>Numerical regression record</h2><p id="verification-status">Checking recorded completion…</p><small>Separate from UI testing. Missing completion does not establish that a process is still running.</small></div><button id="refresh-status" class="secondary">Refresh status</button></section>
</section>
<section id="view-advisory" hidden aria-labelledby="page-title">
<div class="callout"><strong>This view runs the existing policy engine now.</strong> It uses fixed synthetic observations and their scenario clock, not live sensors or Gemma. The UI cannot edit policy coefficients, limits, or execution authority.</div>
<div class="advisory-layout"><section class="card"><h2>Scenario input</h2><label for="scenario">Choose a synthetic condition</label><select id="scenario" disabled></select><p id="scenario-description"></p><div id="scenario-inputs"></div><button id="evaluate" class="primary" disabled>Evaluate scenario</button><p class="small">Advisory only. No equipment actuation or audit-file writes.</p></section><section id="decision" class="card" aria-live="polite"><div class="empty"><span class="eyebrow">READY WHEN YOU ARE</span><h2>Every decision has an execution trace.</h2><p>Select a scenario and evaluate it to inspect the actual rule checks, contributions, and outcome.</p></div></section></div>
</section>
<footer>Bounded evidence, explicit limits. Hash consistency is not biological validation or universal correctness.</footer>
</main></body></html>'''

CSS = r'''
:root{color-scheme:dark;--bg:#0b1016;--panel:#131b25;--panel-2:#182230;--border:#243040;--border-soft:#1c2632;--ink:#e7edf4;--muted:#8ea2b6;--faint:#61758a;--green:#34d399;--green-tint:#0f2b23;--blue:#38bdf8;--blue-tint:#0e2a3c;--amber:#fbbf24;--amber-tint:#37290f;--red:#f87171;--red-tint:#3a1d1b;--gray-tint:#22303d;--ring:#38bdf8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
button,select,input{font:inherit;color:inherit}
button{cursor:pointer}
button:disabled{cursor:wait;opacity:.5}
button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
[hidden]{display:none!important}
::selection{background:#0ea5e94d}
.sidebar{position:fixed;inset:0 auto 0 0;width:235px;background:#0d141d;border-right:1px solid var(--border-soft);color:#dbe7f0;padding:30px 20px;display:flex;flex-direction:column}
.brand{display:flex;gap:12px;align-items:center;font-size:19px;font-weight:700;letter-spacing:-.01em}
.brand-mark{background:linear-gradient(135deg,#34d399,#0ea5e9);color:#06131c;display:grid;place-items:center;width:38px;height:38px;border-radius:10px;font-size:22px;font-weight:800}
.subbrand{display:block;font-size:10px;color:var(--faint);letter-spacing:.16em;text-transform:uppercase;margin-top:3px}
.sidebar-label{font-size:10px;letter-spacing:.16em;color:var(--faint);margin:44px 12px 10px;font-weight:700}
.nav{display:block;width:100%;text-align:left;border:0;background:none;color:#9db0c3;padding:11px 13px;border-radius:9px;margin-bottom:4px;font-size:13.5px;transition:background .12s}
.nav:hover{background:#16202d;color:#dbe7f0}
.nav.active{background:var(--green-tint);color:var(--green);box-shadow:inset 3px 0 var(--green)}
.sidebar-bottom{margin-top:auto;font-size:12px;padding:18px 12px 0;border-top:1px solid var(--border-soft);color:#9db0c3}
.sidebar-bottom div{color:var(--faint);font-size:11px;padding:8px 0 0 15px}
.dot{display:inline-block;width:6px;height:6px;background:var(--green);border-radius:50%;margin-right:8px;box-shadow:0 0 7px #34d39988}
main{margin-left:235px;max-width:1550px;padding:36px 42px}
header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}
h1{font-size:31px;line-height:1.15;letter-spacing:-.03em;margin:6px 0 7px;font-weight:750}
h2{font-size:16.5px;margin:0 0 5px;letter-spacing:-.015em;font-weight:700}
h3{font-size:13px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700}
p{margin:0 0 14px;color:var(--muted)}
header p{margin:0}
.eyebrow{color:var(--green);font-size:10px;font-weight:750;letter-spacing:.17em}
.badge{display:inline-flex;align-items:center;white-space:nowrap;font-size:10px;letter-spacing:.08em;font-weight:750;padding:5px 11px;border-radius:999px;border:1px solid transparent}
.blue{background:var(--blue-tint);color:var(--blue);border-color:#1c435c}
.green{background:var(--green-tint);color:var(--green);border-color:#1d4a3a}
.amber{background:var(--amber-tint);color:var(--amber);border-color:#54401a}
.gray{background:var(--gray-tint);color:#93a7bb;border-color:#33424f}
.red{background:var(--red-tint);color:var(--red);border-color:#552b28}
.notice,.callout{padding:14px 18px;border:1px solid #3a4f2f;background:#16231a;border-radius:10px;font-size:12.5px;margin-bottom:24px;color:#a9c9b0}
.notice strong{color:#d8ecd9}
.callout{font-size:14px;background:#122033;border-color:#1d3a55;color:#a9c6de}
.callout strong{color:#d5e7f5}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:20px}
.metric,.card{background:var(--panel);border:1px solid var(--border-soft);border-radius:13px;box-shadow:0 1px 2px #00000040}
.metric{padding:22px}
.metric span{display:block;font-size:12px;color:var(--muted)}
.metric strong{display:block;font-size:27px;font-weight:700;letter-spacing:-.025em;margin:7px 0;font-variant-numeric:tabular-nums}
.metric small{color:var(--faint);font-size:11px}
.card{padding:24px;margin-bottom:22px}
.section-heading{display:flex;justify-content:space-between;gap:18px}
.section-heading p,.card>p{font-size:12.5px}
.muted,.small,small{font-size:12px;color:var(--muted)}
.layers{display:grid;grid-template-columns:repeat(18,minmax(0,1fr));gap:5px;margin:10px 0 16px}
.layer{padding:12px 1px;border:1px solid var(--border);border-radius:7px;background:#161f2b;color:#71869d;min-width:0;font-size:11px;font-weight:700;transition:transform .1s,border-color .1s}
.layer:hover{transform:translateY(-1px);border-color:#3d5064}
.layer small{font-size:8px;display:block;color:inherit;opacity:.85;letter-spacing:.03em}
.layer.connected{background:#12352a;border-color:#2c6b52;color:#4be0aa}
.layer.partial{background:#3a2c12;border-color:#7a5c22;color:#f3c668}
.legend{display:flex;flex-wrap:wrap;gap:17px;font-size:11px;color:var(--muted);margin-bottom:13px}
.swatch{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px}
.swatch.connected{background:#2f9e75}
.swatch.partial{background:#c99a4b}
.swatch.pending{background:#2a3644}
.evidence-layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:22px;align-items:start}
.experiment{display:block;border:1px solid transparent;background:transparent;text-align:left;padding:12px 13px;border-radius:9px;width:100%;margin:4px 0;color:var(--ink);transition:background .1s}
.experiment strong{display:block;font-size:13px;margin-bottom:6px;font-weight:650}
.experiment:hover{background:#182230}
.experiment.selected{background:#12293a;border-color:#1f4a63}
.divider{border-top:1px solid var(--border-soft);margin:20px 0}
.detail-heading{display:flex;flex-wrap:wrap;gap:10px;align-items:start;justify-content:space-between}
.detail-description{font-size:13px;margin:15px 0}
.mini-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}
.mini-metrics div{padding:13px;border-radius:9px;background:var(--panel-2);border:1px solid var(--border-soft);font-size:11px;color:var(--muted)}
.mini-metrics strong{display:block;font-size:21px;color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums}
.scope{border-left:3px solid #33505f;padding:10px 13px;background:var(--panel-2);border-radius:0 8px 8px 0;font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.instructions{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
.instruction{font:10px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;border:1px solid #1f4a3d;color:#57c99f;background:#0f2620;border-radius:6px;padding:6px 8px;letter-spacing:.02em}
.instruction.selected{background:#0e5f43;color:#eafff6;border-color:#34d399}
.node-detail{font:11px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;background:#0d141d;border:1px solid var(--border-soft);border-radius:9px;padding:13px;margin-bottom:15px;white-space:pre-wrap;color:#a9c4d8}
.hash{font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;color:#7291a8;margin:6px 0 12px}
.hash span{font:700 10px -apple-system,system-ui;display:block;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);margin-bottom:4px}
details{border-top:1px solid var(--border-soft);padding-top:13px;margin-top:15px}
summary{cursor:pointer;font-weight:650;font-size:12.5px;margin-bottom:10px;color:#b7c9da}
summary:hover{color:var(--ink)}
.scroll{max-height:280px;overflow:auto;scrollbar-color:#2c3b4c transparent}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;border-bottom:1px solid var(--border-soft);padding:9px 10px;vertical-align:top;overflow-wrap:anywhere}
th{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:var(--panel-2);position:sticky;top:0}
tbody tr:hover td{background:#ffffff08}
td:last-child{font-variant-numeric:tabular-nums}
.verification{display:flex;justify-content:space-between;gap:20px;align-items:center}
.verification p{margin:3px 0}
.secondary{background:transparent;border:1px solid #33465a;border-radius:8px;padding:8px 14px;color:#c3d3e2;font-size:12.5px;transition:background .1s,border-color .1s}
.secondary:hover{background:#1b2735;border-color:#465d75}
.advisory-layout{display:grid;grid-template-columns:minmax(280px,350px) minmax(0,1fr);gap:22px;align-items:start}
label{display:block;font-size:12px;margin:18px 0 7px;color:var(--muted);font-weight:600}
select{width:100%;background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:14px;color:var(--ink)}
.primary{background:linear-gradient(135deg,#10b981,#0ea5e9);border:0;border-radius:8px;padding:11px 20px;color:#04121c;width:100%;margin:20px 0 10px;font-weight:700;letter-spacing:.01em;transition:filter .12s}
.primary:hover{filter:brightness(1.1)}
.empty{padding:45px 12px}
.empty h2{margin:12px 0}
.result-status{font-size:25px;font-weight:750;letter-spacing:-.02em;margin:10px 0}
.decision-banner{background:#122033;border:1px solid #1d3a55;padding:14px;border-radius:9px;margin:15px 0;font-size:13px;color:#b7cfe2}
.trace-step{border:1px solid var(--border);border-radius:10px;padding:14px;margin:10px 0;background:var(--panel-2)}
.trace-heading{display:flex;gap:10px;justify-content:space-between;align-items:center}
.trace-heading strong{font-size:12.5px}
.trace-step p{font-size:12.5px;margin:8px 0}
.trace-values{font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);margin:0;background:#0d141d;border-radius:7px;padding:10px}
.error{color:#f3b3aa;background:var(--red-tint);border:1px solid #552b28;border-radius:9px;padding:15px;margin-bottom:20px}
.message{padding:18px 0;font-size:14px;color:var(--muted)}
.search{width:100%;padding:8px 10px;background:var(--panel-2);border:1px solid var(--border);border-radius:7px;margin:8px 0;font-size:12.5px;color:var(--ink)}
meter{width:100%;height:22px;accent-color:var(--green)}
footer{font-size:11px;color:var(--faint);padding:10px 0 22px}
.download{margin-top:15px}
.case-row{font-size:12.5px;padding:9px 0;display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--border-soft)}
@media(max-width:1150px){main{padding:28px 23px}
.sidebar{width:200px}
main{margin-left:200px}
.evidence-layout{grid-template-columns:235px minmax(0,1fr)}
.layers{grid-template-columns:repeat(9,minmax(0,1fr))}
.section-heading .muted{display:none}
}
@media(max-width:820px){.sidebar{position:static;width:auto;padding:18px 20px;border-right:0;border-bottom:1px solid var(--border-soft)}
.sidebar nav{display:flex;gap:8px;margin-top:18px}
.sidebar-label,.sidebar-bottom{display:none}
.nav{width:auto;margin:0}
.brand{font-size:18px}
main{margin:0;padding:25px 18px}
.metrics{gap:10px}
.metric{padding:16px}
.metric strong{font-size:21px}
.evidence-layout,.advisory-layout{grid-template-columns:1fr}
.experiments{margin-bottom:0}
#experiments{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.card{padding:20px}
.experiments>.small,.experiments>.divider{display:none}
h1{font-size:27px}
header{align-items:start}
.badge{font-size:9px}
}
@media(max-width:470px){.metrics{grid-template-columns:1fr}
.metric{display:grid;grid-template-columns:1fr auto}
.metric strong{grid-column:2;grid-row:1/3;margin:0}
.metric small{grid-column:1}
.mini-metrics{gap:5px}
.mini-metrics div{padding:9px}
.verification{align-items:start;flex-direction:column}
header{flex-direction:column;gap:14px}
#experiments{grid-template-columns:1fr}
.legend{gap:9px}
.layers{gap:4px}
}
'''

JAVASCRIPT = r'''
'use strict';
const $ = id => document.getElementById(id);
const make = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined && text !== null) node.textContent = String(text); return node; };
const number = value => typeof value === 'number' ? value.toLocaleString() : 'Not recorded';
let selected = 'full-target-holdouts', evidenceRequest, scenarioRequest, scenarioItems = [], selectedNode;
function showError(text) { $('error').textContent = text || ''; $('error').hidden = !text; }
async function api(path, signal) { const result = await fetch(path, {signal, cache: 'no-store'}); if (!result.ok) throw new Error('The local demonstration data could not be loaded (' + result.status + ').'); return result.json(); }
function badge(text, style) { return make('span', 'badge ' + style, text); }
function metric(value, label) { const node = make('div'); node.append(make('strong', '', value), make('span', '', label)); return node; }
function table(headers, rows) { const box = make('div', 'scroll'), t = make('table'), head = make('thead'), hr = make('tr'), body = make('tbody'); headers.forEach(value => hr.append(make('th', '', value))); head.append(hr); rows.forEach(values => { const row = make('tr'); values.forEach(value => row.append(make('td', '', value))); body.append(row); }); t.append(head, body); box.append(t); return box; }
function switchView(advisory) { $('view-advisory').hidden = !advisory; $('view-evidence').hidden = advisory; $('page-title').textContent = advisory ? 'Synthetic advisory' : 'Execution evidence'; $('page-subtitle').textContent = advisory ? 'Run a bounded policy. Inspect the computation behind its decision.' : 'Inspect what was computed, compared, and recorded.'; $('mode-badge').textContent = advisory ? 'LIVE POLICY EVALUATION' : 'SAVED EVIDENCE'; $('mode-badge').className = 'badge ' + (advisory ? 'green' : 'blue'); ['advisory', 'evidence'].forEach(name => { const current = advisory === (name === 'advisory'); $('nav-' + name).classList.toggle('active', current); if (current) $('nav-' + name).setAttribute('aria-current', 'page'); else $('nav-' + name).removeAttribute('aria-current'); }); showError(''); }
$('nav-advisory').addEventListener('click', () => switchView(true));
$('nav-evidence').addEventListener('click', () => switchView(false));
function renderStatus(data) { $('verification-status').textContent = data.complete ? number(data.tests) + ' tests passed in the recorded numerical-stage run.' : data.state === 'failure_recorded' ? 'A failure is recorded. Review the command-line verification log.' : 'Successful completion is not recorded or the log is unavailable.'; }
$('refresh-status').addEventListener('click', async () => { $('refresh-status').disabled = true; try { renderStatus(await api('/api/status')); } catch (error) { showError(error.message); } finally { $('refresh-status').disabled = false; } });
function renderCatalog(data) { $('connected-count').textContent = data.layers.filter(layer => layer.coverage === 'connected_recorded').length + ' / 18 layers'; const full = data.full_target_replay_recorded; const partial = data.layers.find(layer => layer.coverage === 'partial_recorded'); $('frontier').textContent = full ? 'Next-token selection' : partial ? 'Layer ' + partial.index + ' · partial' : 'No partial result shown'; $('frontier-note').textContent = full ? 'Declared boundary/logit comparison and replay passed' : 'Partial evidence is not complete-model execution'; $('layers').replaceChildren(); data.layers.forEach(layer => { const style = layer.coverage === 'connected_recorded' ? 'connected' : layer.coverage === 'partial_recorded' ? 'partial' : ''; const node = make('button', 'layer ' + style, 'L' + layer.index); node.append(make('small', '', layer.attention === 'full' ? 'Full' : 'Local')); node.setAttribute('aria-label', 'Decoder index ' + layer.index + ', ' + layer.coverage.replaceAll('_', ' ')); node.addEventListener('click', () => { $('layer-note').textContent = layer.coverage === 'not_shown' ? 'No independent arithmetic result is displayed for decoder index ' + layer.index + '. This is not a completion estimate.' : layer.coverage === 'partial_recorded' ? 'Layer 2 is only partially covered, using reused boundaries. This is not connected three-layer validation.' : full ? 'All 18 decoder outputs, final normalization, vocabulary logits and token selection matched on the declared cases and replay. This does not expose every internal native operation.' : 'Layers 0 and 1 have connected baseline evidence in saved experiments; separate native forwards were used.'; if (style && layer.evidence_id) selectEvidence(layer.evidence_id); }); $('layers').append(node); }); $('experiments').replaceChildren(); data.experiments.forEach(item => { const button = make('button', 'experiment'); button.dataset.id = item.id; button.append(make('strong', '', item.title), badge(item.availability !== 'available' ? item.availability.toUpperCase() : item.recorded_match ? 'RECORDED MATCH' : 'NO MATCH ASSERTED', item.recorded_match ? 'green' : 'amber')); button.addEventListener('click', () => selectEvidence(item.id)); $('experiments').append(button); }); renderStatus(data.verification); }
function renderEvidence(data) { const box = $('evidence-detail'); box.replaceChildren(); const heading = make('div', 'detail-heading'); heading.append(make('h2', '', data.title), badge(data.recorded_match ? 'RECORDED MATCH' : 'NO MATCH ASSERTED', data.recorded_match ? 'green' : 'amber')); box.append(heading, make('p', 'detail-description', data.description)); if (data.availability !== 'available') { box.append(make('div', 'error', data.message)); return; } const count = data.coverage.instruction_count; const states = data.coverage.new_state_count ?? data.coverage.state_count; const metrics = make('div', 'mini-metrics'); metrics.append(metric(number(count), 'Declared instructions'), metric(number(states), data.coverage.new_state_count !== undefined ? 'New states' : 'States per case'), metric(number(data.mismatches), 'Recorded mismatches')); box.append(metrics, make('div', 'scope', data.scope)); const integrity = make('p', 'small', 'Compact artifacts: pinned and hash-consistent. ' + data.integrity_scope); integrity.classList.add('detail-description'); box.append(integrity); if (data.replay_recorded_match) box.append(badge('RECORDED REPLAY MATCH', 'green')); if (data.cases.length) { box.append(make('h3', '', 'Predeclared cases')); data.cases.forEach(item => { const row = make('div', 'case-row'); row.append(make('span', '', item.id), make('strong', '', number(item.mismatches) + ' recorded mismatches')); box.append(row); if (typeof item.selected_token_id === 'number') box.append(make('p', 'small', 'Predicted/native token: ' + number(item.selected_token_id) + ' · Restored-prefix instructions: prediction ' + number(item.prediction_restored_prefix_count) + ', replay ' + number(item.replay_restored_prefix_count))); }); } box.append(make('h3', '', 'Instruction trace')); if (data.instruction_notice) box.append(make('p', 'small', data.instruction_notice)); const nodes = make('div', 'instructions'), detail = make('div', 'node-detail', 'Select an instruction to inspect its typed inputs and outputs. These are the saved program declarations, not a fresh execution trace.'); selectedNode = null; data.instructions.forEach(instruction => { const node = make('button', 'instruction', instruction.id + ' · ' + instruction.opcode); node.addEventListener('click', () => { if (selectedNode) selectedNode.classList.remove('selected'); node.classList.add('selected'); selectedNode = node; detail.textContent = instruction.id + '  ' + instruction.opcode + '\nLayer: ' + (instruction.layer ?? 'shared/root') + '\nInputs: ' + instruction.inputs.join(', ') + '\nOutputs: ' + instruction.outputs.join(', '); }); nodes.append(node); }); box.append(nodes, detail); const comparisons = make('details'); comparisons.append(make('summary', '', 'Recorded comparison counts (' + data.comparison_counts.length + ' scopes)')); const filter = make('input', 'search'); filter.type = 'search'; filter.placeholder = 'Filter state or comparison scope'; filter.setAttribute('aria-label', 'Filter comparison counts'); const rows = make('div'); const redraw = () => { const search = filter.value.toLowerCase(); rows.replaceChildren(table(['Comparison scope', 'Mismatches'], data.comparison_counts.filter(item => item.name.toLowerCase().includes(search)).map(item => [item.name, number(item.mismatches)]))); }; filter.addEventListener('input', redraw); redraw(); comparisons.append(filter, rows); const hashes = make('details'); hashes.append(make('summary', '', 'Artifact identities and limitations')); Object.entries(data.hashes).forEach(([name, value]) => { const line = make('div', 'hash'); line.append(make('span', '', name.replaceAll('_', ' ')), make('div', '', value)); hashes.append(line); }); hashes.append(make('p', 'small', 'Report hash is a saved reference; the raw report is not checked by this dashboard. Empirical primitive specifications remain in use. No full-model, hardware, or global qualification is granted.')); box.append(comparisons, hashes); }
async function selectEvidence(id) { selected = id; document.querySelectorAll('.experiment').forEach(node => { node.classList.toggle('selected', node.dataset.id === id); node.setAttribute('aria-pressed', String(node.dataset.id === id)); }); if (evidenceRequest) evidenceRequest.abort(); evidenceRequest = new AbortController(); $('evidence-detail').replaceChildren(make('p', 'message', 'Checking saved compact artifacts…')); try { const data = await api('/api/evidence/' + encodeURIComponent(id), evidenceRequest.signal); if (selected === id) renderEvidence(data); } catch (error) { if (error.name !== 'AbortError') { $('evidence-detail').replaceChildren(make('div', 'error', error.message)); } } }
function selectScenario() { if (scenarioRequest) scenarioRequest.abort(); const item = scenarioItems.find(value => value.id === $('scenario').value); if (!item) return; $('scenario-description').textContent = item.description; $('scenario-inputs').replaceChildren(table(['Observation', 'Value'], Object.entries(item.inputs).map(([name, value]) => [name.replaceAll('_', ' '), String(value.value) + ' ' + value.unit])), make('p', 'small', 'Scenario clock: ' + item.evaluated_at)); $('decision').replaceChildren(make('div', 'empty', 'Scenario selected. Evaluate to inspect its actual policy execution.')); $('evaluate').disabled = false; $('evaluate').textContent = 'Evaluate scenario'; }
$('scenario').addEventListener('change', selectScenario);
function renderDecision(data) { const box = $('decision'), decision = data.decision; box.replaceChildren(); box.append(badge('FRESH SYNTHETIC POLICY RESULT', 'blue'), make('div', 'result-status', decision.status), make('p', '', decision.reason), make('p', 'small', data.evaluation_clock)); const status = make('div', 'decision-banner', (data.matches_expected ? 'Matches the scenario expectation. ' : 'Does not match the scenario expectation. ') + (decision.human_review_required ? 'Human review required. ' : '') + 'No equipment action has been taken.'); box.append(status); if (decision.recommendation) { const value = decision.recommendation; box.append(make('h3', '', 'Advisory proposal'), make('p', '', value.field.replaceAll('_', ' ') + ': ' + value.current + ' → ' + value.proposed + ' ' + value.unit + ' (delta ' + value.delta + ')'), badge('AUTHORITY: ADVISORY ONLY', 'amber')); } if (decision.model_output) { const model = decision.model_output; box.append(make('h3', '', 'Transparent model computation'), make('p', 'small', model.formula + ' · score ' + model.score.toFixed(6))); const meter = make('meter'); meter.min = 0; meter.max = 1; meter.value = model.score; meter.setAttribute('aria-label', 'Illustrative model score, not clinical risk'); box.append(meter, table(['Term', 'Contribution to logit'], [['Bias', model.bias], ...Object.entries(model.contributions)].map(([name, value]) => [name.replaceAll('_', ' '), Number(value).toFixed(6)]))); } box.append(make('h3', '', 'Executed checks')); decision.trace.forEach((step, index) => { const card = make('div', 'trace-step'), heading = make('div', 'trace-heading'); heading.append(make('strong', '', (index + 1) + '. ' + step.stage.replaceAll('_', ' ')), badge(step.result, step.result === 'FAIL' ? 'amber' : step.result === 'NO_MATCH' ? 'gray' : 'green')); card.append(heading, make('p', '', step.detail)); const more = make('details'); more.append(make('summary', '', 'Inspect values'), make('pre', 'trace-values', JSON.stringify(step.values, null, 2))); card.append(more); box.append(card); }); const hash = make('div', 'hash'); hash.append(make('span', '', 'Decision identity'), make('div', '', decision.decision_id)); box.append(hash); const download = make('button', 'secondary download', 'Download this decision JSON'); download.addEventListener('click', () => { const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'})); const link = make('a'); link.href = url; link.download = 'synthetic-decision.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }); box.append(download); }
$('evaluate').addEventListener('click', async () => { if (scenarioRequest) scenarioRequest.abort(); const request = new AbortController(); scenarioRequest = request; const id = $('scenario').value; $('evaluate').disabled = true; $('evaluate').textContent = 'Evaluating policy…'; $('decision').replaceChildren(make('p', 'message', 'Running the existing synthetic policy engine…')); try { const data = await api('/api/scenarios/' + encodeURIComponent(id), request.signal); if (scenarioRequest === request) renderDecision(data); } catch (error) { if (error.name !== 'AbortError') $('decision').replaceChildren(make('div', 'error', error.message)); } finally { if (scenarioRequest === request) { $('evaluate').disabled = false; $('evaluate').textContent = 'Evaluate scenario'; } } });
async function initialize() { const results = await Promise.allSettled([api('/api/catalog'), api('/api/scenarios')]); if (results[0].status === 'fulfilled') { renderCatalog(results[0].value); await selectEvidence(selected); } else { showError(results[0].reason.message); $('evidence-detail').replaceChildren(make('p', 'message', 'Evidence catalog unavailable.')); } if (results[1].status === 'fulfilled') { scenarioItems = results[1].value.items; scenarioItems.forEach(item => { const option = make('option', '', item.id.replaceAll('_', ' ')); option.value = item.id; $('scenario').append(option); }); $('scenario').value = scenarioItems.some(item => item.id === 'low_oxygen') ? 'low_oxygen' : scenarioItems[0]?.id || ''; $('scenario').disabled = false; selectScenario(); } else { $('decision').replaceChildren(make('div', 'error', results[1].reason.message)); } }
initialize().catch(error => showError(error.message));
'''
