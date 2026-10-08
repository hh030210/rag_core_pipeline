const $ = selector => document.querySelector(selector);
const el = (tag, className = '', value = null) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== null) node.textContent = String(value);
  return node;
};
let rows = [];
let filtered = [];
let selectedName = null;
let showAll = false;
const detailCache = new Map();
let pendingDetails = new Map();
function readDetail(name) {
  const cachedPayload = window.RETRIEVAL_FUSION_DETAILS?.[name];
  if (cachedPayload) return Promise.resolve(cachedPayload);
  if (pendingDetails.has(name)) return pendingDetails.get(name);
  const path = window.RETRIEVAL_FUSION_INDEX?.detail_files?.[name];
  if (!path) return Promise.reject(new Error(`静态数据包中找不到 ${name}`));

  const promise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = `./${path}`;
    script.onload = () => {
      script.remove();
      const detail = window.RETRIEVAL_FUSION_DETAILS?.[name];
      if (!detail) reject(new Error(`未能加载快照详情：${name}`));
      else {
        delete window.RETRIEVAL_FUSION_DETAILS[name];
        resolve(detail);
      }
    };
    script.onerror = () => {
      script.remove();
      reject(new Error(`读取静态快照失败：${name}`));
    };
    document.head.append(script);
  });
  pendingDetails.set(name, promise);
  promise.then(() => pendingDetails.delete(name), () => pendingDetails.delete(name));
  return promise;
}
function isRank(rank, limit = 5) { return Number.isInteger(rank) && rank > 0 && rank <= limit; }
function state(row) {
  const mapped = Number.isInteger(row.gold_count) && row.gold_count > 0;
  return {
    mapped,
    semHit: mapped && isRank(row.semantic_rank),
    dimHit: mapped && isRank(row.dimension_rank),
    oldHit: mapped && isRank(row.old_rank),
    newHit: mapped && isRank(row.new_rank),
  };
}
function metric(label, value) {
  const card = el('div', 'metric');
  card.append(el('div', 'metric-label', label), el('div', 'metric-value', value));
  return card;
}
function renderMetrics() {
  const mapped = rows.filter(row => state(row).mapped);
  const count = key => mapped.length ? `${mapped.filter(row => state(row)[key]).length} / ${rows.length}` : '未标注';
  $('#metrics').replaceChildren(
    metric('融合快照', rows.length),
    metric('Gold 已映射', `${mapped.length} / ${rows.length}`),
    metric('语义 Top-5', `${count('semHit')}`),
    metric('维度 Top-5', `${count('dimHit')}`),
    metric('原融合 Top-5', `${count('oldHit')}`),
    metric('新融合 Top-5', `${count('newHit')}`),
  );
}
function renderNotes() {
  const mapped = rows.filter(row => state(row).mapped);
  const oldLost = mapped.filter(row => { const s = state(row); return (s.semHit || s.dimHit) && !s.oldHit; }).length;
  const newLost = mapped.filter(row => { const s = state(row); return (s.semHit || s.dimHit) && !s.newHit; }).length;
  const rescued = mapped.filter(row => { const s = state(row); return !s.oldHit && s.newHit; }).length;
  const harmed = mapped.filter(row => { const s = state(row); return s.oldHit && !s.newHit; }).length;
  $('#strategy-notes').hidden = false;
  $('#badcase-summary').textContent = mapped.length
    ? `以 output 中保存的原融合为基准：当前 ${rows.length} 条快照中，单路已进 Top-5、原融合却掉出的查询有 ${oldLost} 条；新融合后还有 ${newLost} 条。新融合救回 ${rescued} 条，同时使 ${harmed} 条原本命中的查询掉出 Top-5。`
    : 'output_new/comparison.jsonl 不存在或没有 Gold 排名；仍可逐条比较原融合与新融合候选。';
}
function passes(row, filter) {
  if (filter === 'all') return true;
  const s = state(row);
  if (!s.mapped) return false;
  if (filter === 'baseline-miss') return !s.oldHit;
  if (filter === 'new-fusion-miss') return !s.newHit;
  if (filter === 'rescued') return !s.oldHit && s.newHit;
  if (filter === 'harmed') return s.oldHit && !s.newHit;
  if (filter === 'semantic-miss') return !s.semHit;
  if (filter === 'dimension-miss') return !s.dimHit;
  if (filter === 'both') return s.semHit && s.dimHit;
  return true;
}
function applyFilters() {
  const text = $('#search').value.trim().toLocaleLowerCase();
  const filter = $('#filter').value;
  filtered = rows.filter(row => (!text || `${row.query} ${row.name}`.toLocaleLowerCase().includes(text)) && passes(row, filter));
  if (!filtered.some(row => row.name === selectedName)) selectedName = filtered[0]?.name || null;
  renderList();
  loadDetail();
}
function renderList() {
  $('#list-count').textContent = `${filtered.length} / ${rows.length}`;
  const list = $('#question-list');
  const cards = filtered.map((row, index) => {
    const button = el('button', `question-item${row.name === selectedName ? ' active' : ''}`);
    button.type = 'button';
    const status = state(row);
    const label = !status.mapped ? 'Gold 未映射' : `新融合 ${status.newHit ? '命中' : '未命中'}`;
    const kind = !status.mapped ? 'unrated' : status.newHit ? 'hit' : 'lost';
    const meta = el('div', 'question-item-meta');
    meta.append(el('span', '', `#${index + 1}`), el('span', `mini-state ${kind}`, label));
    button.append(el('div', 'question-item-title', row.query), meta);
    button.addEventListener('click', () => selectName(row.name));
    return button;
  });
  list.replaceChildren(...cards);
  if (!cards.length) list.append(el('div', 'empty', '没有符合条件的问题'));
}
function selectName(name) { selectedName = name; renderList(); loadDetail(); }
function fmt(value) { return typeof value === 'number' ? value.toFixed(4) : null; }
function goldLabel(rank, mapped) {
  if (!mapped) return 'Gold 排名未提供';
  return Number.isInteger(rank) ? `Golden 最佳名次 #${rank}` : 'Golden 未召回';
}
function renderConstraints(detail) {
  const panel = el('section', 'question-dimensions');
  panel.append(el('h3', 'dimension-panel-title', '问题维度标签'));
  const constraints = detail.query_analysis?.constraints || {};
  const entries = Object.entries(constraints);
  if (!entries.length) {
    panel.append(el('div', 'dimension-note', '当前快照没有保存问题维度解析。'));
    return panel;
  }
  const grid = el('div', 'query-dimension-grid');
  for (const [name, value] of entries) {
    const group = el('div', 'query-dimension');
    group.append(el('div', 'dimension-group-title', `${name} · ${value.role || 'auxiliary'}`));
    const chips = el('div', 'dimension-tags');
    for (const label of value.labels || []) chips.append(el('span', 'dimension-chip', label));
    for (const term of value.intent_terms || []) chips.append(el('span', 'dimension-chip intent', `意图：${term}`));
    if (!chips.childNodes.length) chips.append(el('span', 'dimension-note', '无标签'));
    group.append(chips);
    grid.append(group);
  }
  panel.append(grid);
  return panel;
}
function scoreText(route, item) {
  if (route === 'semantic') return `语义原分 ${fmt(item.semantic_score ?? item.score) || '—'} · 归一化（审计）${fmt(item.normalized_semantic_score) || '—'}`;
  if (route === 'dimension') return `维度原分 ${fmt(item.dimension_score ?? item.score) || '—'} · 归一化（审计）${fmt(item.normalized_dimension_score) || '—'}`;
  if (route === 'old') return `原融合分 ${fmt(item.score) || '—'} · 语义归一化 ${fmt(item.normalized_semantic_score) || '—'} · 维度归一化 ${fmt(item.normalized_dimension_score) || '—'}`;
  const semantic = item.semantic_score_imputed
    ? `语义估计 ${fmt(item.effective_semantic_score) || '—'}（语义路未召回）`
    : `语义原分 ${fmt(item.semantic_score) || '—'}`;
  const contribution = item.score_components;
  const components = contribution
    ? ` · 词面贡献 ${fmt(contribution.lexical) || '—'} · 维度贡献 ${fmt(contribution.dimension) || '—'}`
    : '';
  return `新融合分 ${fmt(item.score) || '—'} · ${semantic} · 词面分 L ${fmt(item.lexical_score) || '—'}${components} · 原融合 #${item.original_fusion_rank ?? '—'}`;
}
function renderFusionDiagnostics(detail) {
  const d = detail.diagnostics || {};
  const panel = el('section', 'question-dimensions');
  panel.append(el('h3', 'dimension-panel-title', '当前查询的融合参数'));
  if (typeof d.semantic_head_gap !== 'number') {
    panel.append(el('div', 'dimension-note', '当前快照未保存头部断层诊断。'));
    return panel;
  }
  const scores = d.semantic_head_scores || [];
  const position = d.semantic_head_gap_position;
  const boundary = Number.isInteger(position) ? `（第 ${position} → ${position + 1} 名）` : '（不足两个语义候选）';
  panel.append(el('p', 'dimension-note', `最大相邻断层 g = ${fmt(d.semantic_head_gap)} ${boundary} · 调节量 u = ${fmt(d.uncertainty)} · 词面权重 λ = ${fmt(d.effective_lexical_weight)} · 维度权重 = ${fmt(d.effective_dimension_weight)}`));
  panel.append(el('p', 'dimension-note', scores.length ? `语义头部原分：${scores.map(fmt).join('、')}` : '语义路未召回候选。'));
  panel.append(el('p', 'dimension-note', `语义路未召回时的估计 S* = ${fmt(d.semantic_missing_estimate)}；候选卡片会标明使用原始分还是估计值。`));
  return panel;
}
function renderMatchInfo(item) {
  const matches = item.matches || [];
  const paths = item.dimension_paths || [];
  if (!matches.length && !paths.length) return null;
  const details = el('details', 'dimension-info');
  details.append(el('summary', '', `维度信息 · ${matches.length} 条匹配`));
  const body = el('div', 'dimension-info-body');
  if (paths.length) body.append(el('div', 'dimension-note', paths.join(' · ')));
  if (matches.length) {
    const list = el('ul', 'dimension-match-list');
    for (const match of matches) {
      const label = [match.dimension_path || match.dimension_id, match.label].filter(Boolean).join(' · ');
      const score = fmt(match.similarity);
      list.append(el('li', '', `${label}${score ? ` · 相似度 ${score}` : ''}`));
    }
    body.append(list);
  }
  details.append(body);
  return details;
}
function showChunk(detail, chunkId, title, rank) {
  const chunk = detail.chunks[chunkId];
  $('#chunk-dialog-title').textContent = chunkId;
  $('#chunk-dialog-meta').textContent = `${title} · 排名 #${rank}${chunk?.title ? ` · ${chunk.title}` : ''}`;
  $('#chunk-content').textContent = chunk?.text || '当前快照没有保存该 chunk 的正文。';
  const dialog = $('#chunk-dialog');
  if (!dialog.open) dialog.showModal();
}
function renderRoute(detail, row, key, title, description, rankKey) {
  const items = detail.routes[key] || [];
  const box = el('section', 'route');
  const head = el('div', 'route-head');
  head.append(el('div', 'route-title', title), el('div', 'route-sub', `${items.length} 个候选 · ${description}`));
  const goldRank = row[rankKey];
  if (state(row).mapped) head.append(el('div', `route-gold-rank${!isRank(goldRank) ? ' lost' : ''}`, goldLabel(goldRank, true)));
  box.append(head);
  const list = el('div', 'route-list');
  const visible = showAll ? items : items.slice(0, 5);
  for (const item of visible) {
    const rank = item.rank;
    const firstGold = state(row).mapped && rank === goldRank;
    const card = el('div', `candidate${firstGold ? ' is-gold' : ''}`);
    card.dataset.chunkId = item.chunk_id;
    card.addEventListener('mouseenter', () => document.querySelectorAll('.candidate[data-chunk-id]').forEach(candidate => candidate.classList.toggle('is-linked-hover', candidate.dataset.chunkId === item.chunk_id)));
    card.addEventListener('mouseleave', () => document.querySelectorAll('.candidate.is-linked-hover').forEach(candidate => candidate.classList.remove('is-linked-hover')));
    const top = el('div', 'candidate-top');
    top.append(el('span', 'rank', `#${rank}`));
    if (firstGold) top.append(el('span', 'flag gold', '首个 GOLDEN'));
    const button = el('button', 'chunk-id chunk-link', item.chunk_id);
    button.type = 'button';
    button.addEventListener('click', () => showChunk(detail, item.chunk_id, title, rank));
    card.append(top, button, el('div', 'score-line', scoreText(key, item)));
    if (key === 'dimension') {
      const info = renderMatchInfo(item);
      if (info) card.append(info);
    }
    list.append(card);
  }
  if (!showAll && items.length > 5) list.append(el('div', 'cutline', `Top-5 截止线 · 后续 ${items.length - 5} 个`));
  if (!items.length) list.append(el('div', 'empty', '该路由没有候选'));
  box.append(list);
  return box;
}
function renderDetail(detail, row) {
  const root = $('#detail');
  const head = el('div', 'detail-top');
  const intro = el('div');
  intro.append(el('h2', 'question-title', row.query));
  intro.append(el('div', 'query-line', `${row.name}${detail.created_at ? ` · ${detail.created_at}` : ''}`));
  const nav = el('div', 'navigation');
  const index = filtered.findIndex(item => item.name === row.name);
  const prev = el('button', 'button', '← 上一题');
  const next = el('button', 'button primary', '下一题 →');
  prev.type = next.type = 'button';
  prev.disabled = index <= 0;
  next.disabled = index >= filtered.length - 1;
  prev.addEventListener('click', () => selectName(filtered[index - 1].name));
  next.addEventListener('click', () => selectName(filtered[index + 1].name));
  nav.append(prev, next);
  head.append(intro, nav);

  const badges = el('div', 'badges');
  for (const [name, key] of [['语义', 'semantic_rank'], ['维度', 'dimension_rank'], ['原融合', 'old_rank'], ['新融合', 'new_rank']]) {
    const rank = row[key];
    const status = !state(row).mapped ? '未标注' : isRank(rank) ? `Top-5 命中 #${rank}` : Number.isInteger(rank) ? `第 ${rank} 名` : '未召回';
    badges.append(el('span', `badge${state(row).mapped ? isRank(rank) ? ' hit' : ' miss' : ''}`, `${name} ${status}`));
  }
  badges.append(el('span', 'badge gold', state(row).mapped ? `Golden ${row.gold_count} 个` : 'Gold 未映射'));
  const routes = el('div', 'routes');
  routes.append(
    renderRoute(detail, row, 'semantic', '语义检索', '相似度排序', 'semantic_rank'),
    renderRoute(detail, row, 'dimension', '维度检索', '维度匹配排序', 'dimension_rank'),
    renderRoute(detail, row, 'old', '原融合', '来自 output', 'old_rank'),
    renderRoute(detail, row, 'new', '新融合', '原始语义分 + 动态词面贡献', 'new_rank'),
  );
  const foot = el('div', 'detail-foot');
  foot.append(el('span', 'chunk-hint', '点击 chunk 编号查看快照中保存的正文。'));
  if (state(row).mapped) foot.append(el('span', '', 'Gold 标记显示各路首个 golden；排名统计来自 output_new/comparison.jsonl。'));
  root.replaceChildren(head, badges, renderFusionDiagnostics(detail), renderConstraints(detail), routes, foot);
}
async function loadDetail() {
  const root = $('#detail');
  const row = filtered.find(item => item.name === selectedName);
  if (!row) { root.replaceChildren(el('div', 'empty', '没有符合条件的问题')); return; }
  const name = row.name;
  if (detailCache.has(name)) { renderDetail(detailCache.get(name), row); return; }
  root.replaceChildren(el('div', 'empty', '正在读取所选文件夹中的快照…'));
  try {
    const detail = await readDetail(name);
    if (detailCache.size >= 8) detailCache.delete(detailCache.keys().next().value);
    detailCache.set(name, detail);
    if (!comparisonsByName.has(name) && detail.query) {
      row.query = detail.query;
      renderList();
    }
    if (selectedName === name) renderDetail(detail, row);
  } catch (error) {
    if (selectedName === name) root.replaceChildren(el('div', 'error', `读取快照失败：${error.message}`));
  }
}
function bindInteractions() {
  $('#search').addEventListener('input', applyFilters);
  $('#filter').addEventListener('change', applyFilters);
  $('#show-all').addEventListener('click', () => {
    showAll = !showAll;
    $('#show-all').textContent = showAll ? '仅显示 Top-5' : '展开全部候选';
    loadDetail();
  });
  $('#chunk-dialog-close').addEventListener('click', () => $('#chunk-dialog').close());
}
function start() {
  bindInteractions();
  const source = window.RETRIEVAL_FUSION_INDEX;
  if (!Array.isArray(source?.items) || !source.items.length) {
    $('#app').replaceChildren(el('div', 'error', '没有加载到静态快照数据，请运行 build_static_data.py 生成数据文件。'));
    return;
  }
  rows = source.items;
  $('#subtitle').textContent = '静态快照已内置在页面目录中；点击 chunk 编号查看正文。';
  renderMetrics();
  renderNotes();
  const hasGold = rows.some(row => state(row).mapped);
  for (const option of $('#filter').options) if (option.value !== 'all') option.disabled = !hasGold;
  applyFilters();
}
start();
