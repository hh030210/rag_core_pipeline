const $ = selector => document.querySelector(selector);
const el = (tag, className = '', value = null) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== null) node.textContent = String(value);
  return node;
};
let rows = [];
let rowsByStrategy = {};
let liveAccumulators = {};
let vectorStrategiesBySource = {};
let fusionStrategiesBySource = {};
let currentPipelineStrategies = {vector:null, fusion:null};
let adaptiveManifest = null, adaptiveAccumulator = null;
let filtered = [];
let selectedName = null;
const topK = { semantic: 5, dimension: 5, fusion: 5 };
const caseTopK = { semantic: 5, dimension: 5, fusion: 5 };
let caseStrategy = 'recommended';
const chartResizeObserver = new ResizeObserver(entries => {
  for (const { target } of entries) {
    for (const segment of target.querySelectorAll('.fusion-band')) {
      const label = segment.querySelector('.fusion-band-label');
      segment.classList.toggle('compact', segment.getBoundingClientRect().width < label.scrollWidth + 8);
    }
  }
});
const detailCache = new Map();
let manifest = null;
let dimensionManifest = null;
let recommendedManifest = null;
const DATASETS = {
  adaptive: '../experiments/00_method_comparison/output/online_adaptive/dataset/',
  online: '../experiments/00_method_comparison/output/online/dataset/',
  dimension: '../experiments/00_method_comparison/output/dimension_score/dataset/',
  recommended: '../experiments/00_method_comparison/output/recommended_fusion/dataset/',
};
let activeStrategy = 'recommended';
function strategyIndex(strategy) { if (strategy === 'adaptive') return adaptiveManifest; return strategy === 'old' ? manifest : strategy === 'new' ? dimensionManifest : recommendedManifest; }
function strategyRows(strategy) { return importedDetails ? rows : rowsByStrategy[strategy] || []; }
function batchLabel(strategy) {
  const index = strategyIndex(strategy);
  if (!index) return '尚未运行';
  const date = index.generated_at ? new Date(index.generated_at).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai'}) : '运行时间未记录';
  return date + ' · ' + Object.keys(index.detail_files).length + ' 条查询';
}
function offlineLabel() {
  return strategyLabel(activeStrategy);
}
function strategyLabel(strategy) {
  if (strategy === 'adaptive') return '自适应融合基准';
  return strategy === 'recommended' ? '推荐融合'
    : strategy === 'old' ? '线上融合' : '离线融合（维度分数基准）';
}
function isRank(rank, limit = 5) { return Number.isInteger(rank) && rank > 0 && rank <= limit; }
function state(row, strategy = 'recommended', limits = { semantic: 5, dimension: 5, fusion: 5 }) {
  const mapped = Number.isInteger(row.gold_count) && row.gold_count > 0;
  return {
    mapped,
    semHit: mapped && isRank(row.semantic_rank, limits.semantic),
    dimHit: mapped && isRank(row.dimension_rank, limits.dimension),
    oldHit: mapped && isRank(row.old_rank, limits.fusion),
    newHit: mapped && isRank(row[`${strategy}_rank`], limits.fusion),
  };
}
function metric(label, value) {
  const card = el('div', 'metric');
  card.append(el('div', 'metric-label', label), el('div', 'metric-value', value));
  return card;
}
function renderMetrics(evaluation) {
  const mapped = evaluation?.mapped_queries ?? rows.filter(row => state(row).mapped).length;
  const unmapped = evaluation?.unmapped_queries ?? rows.length - mapped;
  const total = evaluation?.total_queries ?? rows.length;
  $('#metrics').replaceChildren(
    metric('线上评测查询', total),
    metric('Gold 已映射', `${mapped} / ${total}`),
    metric('未映射', unmapped),
  );
}
function renderEvaluation(evaluation) {
  const panel = $('#evaluation-panel');
  const metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'Hit@15', 'MRR@10', 'nDCG@5'];
  if (!evaluation?.metrics?.length) { panel.hidden = true; return; }
  const mode = $('#metric-denominator').value;
  const groups = importedDetails ? {old: evaluation}
    : Object.fromEntries(Object.entries(liveAccumulators).map(([key, acc]) => [key, acc.result(mode)]));
  if (adaptiveAccumulator && !importedDetails) groups.adaptive = adaptiveAccumulator.result(mode);
  const definitions = [['semantic', 'old'], ['dimension', 'old'], ['old', 'adaptive'], ['new', 'new'], ['recommended', 'recommended']];
  const entries = definitions.flatMap(([key, source]) => {
    const group = groups[source];
    const metric = group?.metrics.find(row => row.key === key);
    return metric ? [{...metric, group, source}] : [];
  });
  if (importedDetails) {
    entries.splice(0, entries.length, ...evaluation.metrics.map(row => ({...row, group:evaluation, source:'old'})));
  }
  const head = el('div', 'evaluation-head');
  const title = el('h2', '', '融合策略对比（根据候选现算）');
  title.id = 'evaluation-title'; head.append(title);
  head.append(el('div', 'evaluation-note', '展示语义、维度路线及 code 中全部三种融合策略；各策略读取自己的候选与 Golden。'));
  panel.replaceChildren(head);
  const strategyNames = vectorStrategiesBySource.old || [];
  panel.append(el('p', 'evaluation-note', '本批向量策略：' + (strategyNames.join('、') || '未记录（旧批次）')
    + ' · 当前主线配置：' + (currentPipelineStrategies.fusion || '读取中')
    + ' · 本批线上融合：' + (fusionStrategiesBySource.old || []).join('、')
    + ' · 查询 ' + evaluation.total_queries + ' · Gold 已映射 ' + evaluation.mapped_queries
    + ' · 指标分母 ' + (mode === 'all' ? '全部查询' : 'Gold 已映射查询')));
  if (importedDetails) {
    panel.append(el('p', 'evaluation-note', metricSource));
  } else {
    panel.append(el('p', 'evaluation-note', ['adaptive', 'new', 'recommended'].filter(key => groups[key])
      .map(key => strategyLabel(key) + '：' + batchLabel(key)).join('；')));
    const stale = (adaptiveManifest && adaptiveManifest.input_batch_id !== manifest.batch_id) || (dimensionManifest && dimensionManifest.input_batch_id !== manifest.batch_id)
      || (recommendedManifest && recommendedManifest.input_batch_id !== dimensionManifest?.batch_id);
    if (stale) panel.append(el('p', 'evaluation-note', '注意：06 或 07 尚未跟随当前线上批次更新，请运行 run_all.py。各行仍按自己的实验数据计算。'));
  }
  const table = el('table', 'evaluation-table'), thead = el('thead'), header = el('tr');
  header.append(el('th', '', '检索路线'));
  for (const label of metrics) header.append(el('th', '', label + ' (%)'));
  thead.append(header);
  const tbody = el('tbody');
  for (const row of entries) {
    const tr = el('tr'); tr.dataset.route = row.key; tr.dataset.source = row.source;
    if (row.key === 'recommended') tr.className = 'is-new';
    const filename = {adaptive:'online_adaptive.py', new:'dimension_score.py', recommended:'recommended_fusion.py'}[row.source];
    const heading = el('th', '', row.source === 'adaptive' ? '自适应融合基准' : row.label); heading.scope = 'row';
    if (filename) { heading.append(el('code','strategy-filename',filename)); tr.dataset.strategyFile=filename;
      if (filename === currentPipelineStrategies.fusion) { heading.append(el('span','mainline-badge','当前主线'));tr.classList.add('is-mainline'); } }
    if (row.key === 'semantic') heading.title = strategyNames.join('、');
    tr.append(heading);
    for (const label of metrics) {
      const td = el('td');
      td.textContent = row.complete && row.group.denominator
        ? label.startsWith('Hit@')
          ? (row.hits[label] / row.group.denominator * 100).toFixed(2) + '% (' + row.hits[label] + '/' + row.group.denominator + ')'
          : (row.ranking_scores[label] * 100).toFixed(2) + '%'
        : '—';
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(thead, tbody); const wrap = el('div', 'evaluation-table-wrap'); wrap.append(table); panel.append(wrap);
  panel.hidden = false;
}
function renderRouteFusionChart() {
  const categories = [
    { key: 'both', label: '语义命中 · 维度命中', sem: true, dim: true },
    { key: 'semantic', label: '语义命中 · 维度未命中', sem: true, dim: false },
    { key: 'dimension', label: '语义未命中 · 维度命中', sem: false, dim: true },
    { key: 'neither', label: '语义未命中 · 维度未命中', sem: false, dim: false },
  ];
  const mapped = strategyRows(activeStrategy).filter(row => state(row, activeStrategy, topK).mapped);
  const groups = categories.map(category => mapped.filter(row => {
    const s = state(row, activeStrategy, topK);
    return s.semHit === category.sem && s.dimHit === category.dim;
  }));
  const maxCount = Math.max(1, ...groups.map(group => group.length));
  const head = el('div', 'route-fusion-head');
  const title = el('h2', '', '融合策略分析 · 两路命中组合');
  title.id = 'route-fusion-title';
  head.append(title, el('div', 'route-fusion-note',
    offlineLabel() + ' · ' + batchLabel(activeStrategy) + ' · 语义 Top-' + topK.semantic + ' / 维度 Top-' + topK.dimension + ' / 融合 Top-' + topK.fusion + ' · Gold 已映射 ' + mapped.length + ' 条'));
  const table = el('table', 'fusion-table');
  const thead = el('thead'), header = el('tr');
  for (const label of ['两路检索命中情况', '查询总数', '融合结果 · 查询数量', '命中率']) header.append(el('th', '', label));
  thead.append(header);
  const tbody = el('tbody');
  for (const [index, category] of categories.entries()) {
    const group = groups[index];
    const hits = group.filter(row => state(row, activeStrategy, topK).newHit).length;
    const misses = group.length - hits;
    const tr = el('tr'); tr.dataset.category = category.key;
    const label = el('th', '', category.label); label.scope = 'row';
    tr.append(label, el('td', 'fusion-total', group.length));
    const result = el('td', 'fusion-result');
    const scale = el('div', 'fusion-scale');
    const track = el('div', 'fusion-rate-track');
    track.style.width = `${group.length / maxCount * 100}%`;
    track.setAttribute('role', 'group');
    track.setAttribute('aria-label', '融合命中 ' + hits + ' 条，未命中 ' + misses + ' 条');
    for (const [count, kind, label] of [[hits, 'hit', '融合命中'], [misses, 'miss', '融合未命中']]) {
      if (!count) continue;
      const segment = el('button', `fusion-band ${kind}`);
      segment.type = 'button';
      const filter = `combination-${Number(category.sem)}${Number(category.dim)}${kind === 'hit' ? '1' : '0'}`;
      segment.dataset.filter = filter;
      segment.setAttribute('aria-label', `${category.label} · ${label}，${count} 条，点击查看 ${offlineLabel()} 对应详情`);
      segment.addEventListener('click', () => showChartCases(filter));
      segment.append(el('span', 'fusion-band-label', count));
      segment.style.flexGrow = String(count);
      segment.dataset.count = String(count);
      segment.title = `${category.label} · ${label} ${count} 条，点击查看详情`;
      track.append(segment);
    }
    scale.append(track);
    if (!group.length) scale.append(el('span', 'fusion-band-empty', '无查询'));
    result.append(scale);
    tr.append(result, el('td', 'fusion-rate', group.length ? (hits / group.length * 100).toFixed(1) + '%' : '—'));
    tbody.append(tr);
  }
  table.append(thead, tbody);
  const wrap = el('div', 'evaluation-table-wrap'); wrap.append(table);
  const hits = mapped.filter(row => state(row, activeStrategy, topK).newHit).length;
  const legend = el('div', 'fusion-band-legend');
  for (const [kind, label] of [['hit', '融合命中'], ['miss', '融合未命中']]) {
    const item = el('span');
    item.append(el('i', `fusion-band-swatch ${kind}`), document.createTextNode(label));
    legend.append(item);
  }
  $('#route-fusion-chart').replaceChildren(head, legend, wrap,
    el('div', 'route-fusion-note', '合计：融合命中 ' + hits + ' 条，未命中 ' + (mapped.length - hits) + ' 条。点击色段查看对应类别详情。条带长度与数量成正比；窄色段数字显示在相邻上方或下方。未映射问题不参与统计。'));
  chartResizeObserver.disconnect();
  for (const track of wrap.querySelectorAll('.fusion-rate-track')) chartResizeObserver.observe(track);
}
function showChartCases(filter) {
  caseStrategy = activeStrategy;
  $('#case-strategy-select').value = caseStrategy;
  for (const key of ['semantic', 'dimension', 'fusion']) {
    const value = topK[key];
    caseTopK[key] = value;
    const select = $(`#case-${key}-topk`);
    if (![...select.options].some(option => Number(option.value) === value)) {
      const option = el('option', '', value);
      option.value = String(value);
      const next = [...select.options].find(item => Number(item.value) > value);
      select.add(option, next || null);
    }
    select.value = String(value);
  }
  $('#search').value = '';
  $('#filter').value = filter;
  selectedName = null;
  refreshCasePanel();
  $('#filter').focus({ preventScroll: true });
  $('#case-panel').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
}
function passes(row, filter) {
  if (filter === 'all') return true;
  const s = state(row, caseStrategy, caseTopK);
  if (filter === 'unmapped') return !s.mapped;
  if (!s.mapped) return false;
  if (filter === 'fusion-miss') return !s.newHit;
  if (filter === 'semantic-miss') return !s.semHit;
  if (filter === 'dimension-miss') return !s.dimHit;
  const combination = /^combination-([01])([01])([01])$/.exec(filter);
  if (combination) return [s.semHit, s.dimHit, s.newHit].every((hit, index) => hit === (combination[index + 1] === '1'));
  return true;
}
function refreshCasePanel() {
  const labels = {
    'semantic-miss': `语义 Top-${caseTopK.semantic} 未命中`,
    'dimension-miss': `维度 Top-${caseTopK.dimension} 未命中`,
    'fusion-miss': `融合 Top-${caseTopK.fusion} 未命中`,
  };
  for (const option of $('#filter').options) if (labels[option.value]) option.textContent = labels[option.value];
  $('#case-settings-note').textContent = `仅影响本 panel：${strategyLabel(caseStrategy)}，语义 Top-${caseTopK.semantic} / 维度 Top-${caseTopK.dimension} / 融合 Top-${caseTopK.fusion}。使用该实验独立保存的输入、Golden 和结果。${batchLabel(caseStrategy)}`;
  const sourceRows = strategyRows(caseStrategy);
  const mapped = sourceRows.some(row => state(row).mapped);
  for (const option of $('#filter').options) if (option.value !== 'all') option.disabled = !mapped && option.value !== 'unmapped';
  $('#filter').querySelector('option[value="unmapped"]').textContent = '未映射（' + sourceRows.filter(row => !state(row).mapped).length + '）';
  applyFilters();
}
function applyFilters() {
  const text = $('#search').value.trim().toLocaleLowerCase();
  const filter = $('#filter').value;
  filtered = strategyRows(caseStrategy).filter(row => (!text || `${row.query} ${row.name}`.toLocaleLowerCase().includes(text)) && passes(row, filter));
  if (!filtered.some(row => row.name === selectedName)) selectedName = filtered[0]?.name || null;
  renderList();
  loadDetail();
}
function renderList() {
  $('#list-count').textContent = `${filtered.length} / ${strategyRows(caseStrategy).length}`;
  const list = $('#question-list');
  const cards = filtered.map((row, index) => {
    const button = el('button', `question-item${row.name === selectedName ? ' active' : ''}`);
    button.type = 'button';
    const status = state(row, caseStrategy, caseTopK);
    const label = !status.mapped ? '未映射 · 不计 Gold 指标' : `${strategyLabel(caseStrategy)} Top-${caseTopK.fusion} ${status.newHit ? '命中' : '未命中'}`;
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
  if (route === 'semantic') return `语义分 ${fmt(item.semantic_score ?? item.score) || '—'} · 归一化 ${fmt(item.normalized_semantic_score) || '—'}`;
  if (route === 'dimension') return `维度分 ${fmt(item.dimension_score ?? item.score) || '—'} · 归一化 ${fmt(item.normalized_dimension_score) || '—'}`;
  if (route === 'old') return `原融合分 ${fmt(item.score) || '—'} · 语义 ${fmt(item.normalized_semantic_score) || '—'} · 维度 ${fmt(item.normalized_dimension_score) || '—'}`;
  if (item.score_components) {
    const parts = Object.entries(item.score_components)
      .map(([name, value]) => `${{semantic: '语义贡献', dimension: '维度贡献', lexical: '词面贡献',
        window_replacement: '局部证据修正',
        dimension_rescale: '维度权重修正', lexical_rescale: '词面权重修正',
        semantic_rank_adjustment: '语义名次修正',
        intent_anchor_bonus: '事实锚点加分'}[name] || name} ${fmt(value) || '0.0000'}`).join(' · ');
    const dim = typeof item.effective_dimension_score === 'number' ? ` · 使用的维度分 ${fmt(item.effective_dimension_score)}` : '';
    const imputed = item.semantic_score_imputed ? ' · 语义分为缺失估计' : '';
    return `融合分 ${fmt(item.score) || '—'} · ${parts}${dim}${imputed}`;
  }
  return `新融合分 ${fmt(item.score) || '—'} · 词组 ${fmt(item.lexical_score) || '—'} · 原融合 #${item.original_fusion_rank ?? '—'}`;
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
  const limit = caseTopK[key] ?? caseTopK.fusion;
  const box = el('section', 'route');
  const head = el('div', 'route-head');
  head.append(el('div', 'route-title', title), el('div', 'route-sub', `${description} · 全部 ${items.length} 个候选`));
  const goldRank = row[rankKey];
  if (state(row).mapped) head.append(el('div', `route-gold-rank${!isRank(goldRank, limit) ? ' lost' : ''}`, `${goldLabel(goldRank, true)} · Top-${limit} ${isRank(goldRank, limit) ? '命中' : '未命中'}`));
  box.append(head);
  const list = el('div', 'route-list');
  list.tabIndex = 0;
  list.setAttribute('aria-label', `${title} 全部候选，可滚动查看`);
  for (const item of items) {
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
  for (const [name, key] of [['语义', 'semantic_rank'], ['维度', 'dimension_rank'], [caseStrategy === 'adaptive' ? '自适应融合' : '主线评测', 'old_rank'], ...(detail.routes.new ? [['离线融合', 'new_rank']] : []), ...(detail.recommended ? [['推荐融合', 'recommended_rank']] : [])]) {
    const limit = key === 'semantic_rank' ? caseTopK.semantic : key === 'dimension_rank' ? caseTopK.dimension : caseTopK.fusion;
    const rank = row[key], hit = isRank(rank, limit);
    const status = !state(row).mapped ? '未标注' : hit ? `Top-${limit} 命中 #${rank}` : Number.isInteger(rank) ? '第 ' + rank + ' 名' : '未召回';
    badges.append(el('span', 'badge' + (state(row).mapped ? hit ? ' hit' : ' miss' : ''), name + ' ' + status));
  }
  badges.append(el('span', 'badge gold', state(row).mapped ? 'Golden ' + row.gold_count + ' 个' : 'Gold 未映射'));
  const routes = el('div', 'routes');
  const routeDetail = {...detail, routes: {...detail.routes, recommended: detail.recommended?.candidates || []}};
  routes.append(
    renderRoute(detail, row, 'semantic', '语义检索', '相似度排序', 'semantic_rank'),
    renderRoute(detail, row, 'dimension', '维度检索', '维度匹配排序', 'dimension_rank'),
    renderRoute(detail, row, 'old', caseStrategy === 'adaptive' ? '自适应融合基准' : '主线评测结果', caseStrategy === 'adaptive' ? 'online_adaptive.py' : '已保存的线上排序', 'old_rank'),
    ...(detail.routes.new ? [renderRoute(detail, row, 'new', '离线融合', '维度分数融合', 'new_rank')] : []),
    ...(detail.recommended ? [renderRoute(routeDetail, row, 'recommended', '推荐融合', '维度角色校准', 'recommended_rank')] : []),
  );
  renderRecommended(detail, row, routes);
  const foot = el('div', 'detail-foot');
  foot.append(el('span', 'chunk-hint', '点击 chunk 编号查看快照中保存的正文。'));
  foot.append(el('span', '', '各列可独立滚动查看全部已保存候选；命中标记采用本 panel 的 Top-K，融合筛选采用本 panel 所选策略。'));
  root.replaceChildren(head, badges, renderConstraints(detail), routes, foot);
}
async function loadDetail() {
  const root = $('#detail');
  const row = filtered.find(item => item.name === selectedName);
  if (!row) { root.replaceChildren(el('div', 'empty', '没有符合条件的问题')); return; }
  const name = row.name;
  const sourceStrategy = caseStrategy;
  const cacheKey = sourceStrategy + ':' + name;
  root.replaceChildren(el('div', 'empty', '正在读取详情…'));
  try {
    let detail = importedDetails?.get(name) || detailCache.get(cacheKey);
    if (!detail) detail = await loadExperimentDetail(name, sourceStrategy);
    if (detailCache.size >= 8) detailCache.delete(detailCache.keys().next().value);
    detailCache.set(cacheKey, detail);
    if (selectedName === name && sourceStrategy === caseStrategy) renderDetail(detail, row);
  } catch (error) {
    if (selectedName === name) root.replaceChildren(el('div', 'error', '读取详情失败：' + error.message));
  }
}
async function start() {
  try {
    if (location.protocol === 'file:') throw new Error('请通过 HTTP 服务打开页面；页面直接读取 experiments 中的 JSON，打开方式见 README.md');
    await loadLatestExperimentIndexes();
    $('#strategy-select').addEventListener('change', () => { activeStrategy = $('#strategy-select').value; renderRouteFusionChart(); });
    for (const key of ['semantic', 'dimension', 'fusion']) {
      $('#' + key + '-topk').addEventListener('change', event => {
        const value = Number(event.target.value);
        if (!Number.isSafeInteger(value) || value < 1) { event.target.value = topK[key]; return; }
        topK[key] = value; renderRouteFusionChart();
      });
    }
    $('#subtitle').textContent = '各实验独立保存输入与结果；指标、命中图和详情按所选实验的数据计算。';
    $('#metric-denominator').addEventListener('change', refreshCalculatedTable);
    $('#recalculate-metrics').addEventListener('click', reloadLatestMetrics);
    $('#import-evaluation').addEventListener('change', async event => {
      const file = event.target.files[0]; if (!file) return;
      try { await importEvaluation(file); } catch (error) { $('#metric-status').textContent = '导入失败：' + error.message; }
      event.target.value = '';
    });
    $('#restore-snapshots').addEventListener('click', () => location.reload());
    const hasGold = rows.some(row => state(row).mapped);
    const unmappedCount = rows.filter(row => !state(row).mapped).length;
    const unmappedOption = $('#filter').querySelector('option[value="unmapped"]');
    if (unmappedOption) unmappedOption.textContent = `未映射（${unmappedCount}）`;
    for (const option of $('#filter').options) {
      if (option.value !== 'all') option.disabled = !hasGold && option.value !== 'unmapped';
    }
    $('#search').addEventListener('input', applyFilters);
    $('#filter').addEventListener('change', applyFilters);
    $('#case-strategy-select').addEventListener('change', event => {
      caseStrategy = event.target.value;
      refreshCasePanel();
    });
    for (const key of ['semantic', 'dimension', 'fusion']) {
      const select = $(`#case-${key}-topk`);
      select.replaceChildren(...[1, 3, 5, 10, 15, 20, 50, 100].map(value => {
        const option = el('option', '', value);
        option.value = String(value);
        return option;
      }));
      select.value = String(caseTopK[key]);
      select.addEventListener('change', event => {
        caseTopK[key] = Number(event.target.value);
        refreshCasePanel();
      });
    }
    $('#chunk-dialog-close').addEventListener('click', () => $('#chunk-dialog').close());
    await calculateLiveMetrics();
  } catch (error) {
    $('#app').replaceChildren(el('div', 'error', `未能加载融合快照：${error.message}`));
  }
}

const rawPayloadLoads = new Map();
let importedDetails = null;
let liveAccumulator = null;
let metricGeneration = 0;
let metricSource = 'experiments 实验基准';
function refreshCalculatedTable() {
  if (!liveAccumulator) return;
  const evaluation = liveAccumulator.result($('#metric-denominator').value);
  renderMetrics(evaluation);
  renderEvaluation(evaluation);
}
async function readJSON(path) {
  const url = new URL(path, location.href).href;
  if (rawPayloadLoads.has(url)) return rawPayloadLoads.get(url);
  const requestURL = new URL(url);
  requestURL.searchParams.set('_fresh', String(Date.now()));
  const load = fetch(requestURL.href, {cache: 'no-store'}).then(response => {
    if (!response.ok) throw new Error('无法读取实验数据：' + path + '（HTTP ' + response.status + '）');
    return response.json();
  }).finally(() => rawPayloadLoads.delete(url));
  rawPayloadLoads.set(url, load);
  return load;
}
async function readExperimentIndex(base) {
  const index = await readJSON(base + 'index.json');
  if (!index.detail_files || typeof index.detail_files !== 'object') throw new Error('实验索引缺少 detail_files：' + base);
  index.detail_files = Object.fromEntries(Object.entries(index.detail_files).map(([name, path]) => [name, base + path]));
  return index;
}
async function loadLatestExperimentIndexes() {
  const optionalIndex = async base => { try { return await readExperimentIndex(base); } catch (error) { if (error.message.includes('HTTP 404')) return null; throw error; } };
  [manifest, dimensionManifest, recommendedManifest, adaptiveManifest] = await Promise.all([
    readExperimentIndex(DATASETS.online), optionalIndex(DATASETS.dimension), optionalIndex(DATASETS.recommended), readExperimentIndex(DATASETS.adaptive),
  ]);
  if (!Array.isArray(manifest.items) || !manifest.items.length) throw new Error('线上评测索引为空');
  for (const strategy of ['old', 'adaptive', 'new', 'recommended']) {
    const index = strategyIndex(strategy);
    if (index && !Array.isArray(index.items)) index.items = Object.keys(index.detail_files).map(name => ({name}));
    for (const id of ['strategy-select', 'case-strategy-select']) $('#' + id).querySelector('option[value="' + strategy + '"]').disabled = !index;
  }
  if (!strategyIndex(activeStrategy)) activeStrategy = 'adaptive';
  if (!strategyIndex(caseStrategy)) caseStrategy = 'adaptive';
  $('#strategy-select').value = activeStrategy;
  $('#case-strategy-select').value = caseStrategy;
  metricSource = '主线批次与各融合方法分别读取实验数据';
  rows = manifest.items.map(row => ({...row}));
}
async function reloadLatestMetrics() {
  try {
    if (!importedDetails) {
      await loadLatestExperimentIndexes();
      detailCache.clear();
      selectedName = null;
    }
    await calculateLiveMetrics();
  } catch (error) {
    liveAccumulator = null;
    $('#evaluation-panel').replaceChildren(el('div', 'error', '读取最新评测失败：' + error.message));
    $('#metric-status').textContent = '读取失败；未回退旧数据';
  }
}
async function loadExperimentDetail(name, strategy = 'old') {
  const index = strategyIndex(strategy), path = index?.detail_files[name];
  if (!path) throw new Error('该实验缺少查询：' + name);
  const detail = await readJSON(path);
  if (detail.name !== name || detail.batch_id !== index.batch_id || !detail.routes || !detail.gold || !detail.chunks) throw new Error('该实验的输入快照或批次不完整：' + name);
  if (strategy === 'new') detail.routes.new = detail.fusion_candidates;
  if (strategy === 'recommended') detail.recommended = {candidates: detail.candidates, diagnostics: detail.diagnostics, configuration: detail.configuration};
  return detail;
}
async function calculateLiveMetrics() {
  const generation = ++metricGeneration;
  $('#recalculate-metrics').disabled = true;
  $('#evaluation-panel').hidden = false;
  $('#evaluation-panel').replaceChildren(el('div', 'evaluation-note', '正在分别计算各实验候选与 Golden…'));
  try {
    const nextRows = {}, accumulators = {}, nextVectorStrategies = {}, nextFusionStrategies = {};
    const strategies = importedDetails ? ['old'] : ['old', 'new', 'recommended'].filter(strategy => strategyIndex(strategy));
    const total = strategies.reduce((n, strategy) => n + (importedDetails ? rows.length : strategyIndex(strategy).items.length), 0);
    let completed = 0;
    for (const strategy of strategies) {
      const acc = RetrievalMetrics.accumulator();
      const vectorNames = new Set();
      const fusionNames = new Set();
      const updated = (importedDetails ? rows : strategyIndex(strategy).items).map(row => ({...row}));
      for (let offset = 0; offset < updated.length; offset += 4) {
        await Promise.all(updated.slice(offset, offset + 4).map(async row => {
          const detail = importedDetails?.get(row.name) || await loadExperimentDetail(row.name, strategy);
          Object.assign(row, acc.add(detail));
          vectorNames.add(detail.query_analysis?.vector_retrieval_strategy || "未记录（旧批次）");
          fusionNames.add(detail.query_analysis?.fusion_strategy || detail.evaluation?.retrieval?.fusion_strategy?.file || detail.evaluation?.retrieval?.fusion_strategy?.name || 'adaptive_report_v1（迁移前基准）');
          row.query = detail.query;
        }));
        if (generation !== metricGeneration) return;
        completed += Math.min(4, updated.length - offset);
        $('#metric-status').textContent = '独立计算中 ' + completed + '/' + total;
      }
      nextRows[strategy] = updated;
      accumulators[strategy] = acc;
      nextVectorStrategies[strategy] = [...vectorNames];
      nextFusionStrategies[strategy] = [...fusionNames];
    }
    rowsByStrategy = nextRows;
    rows = nextRows.old;
    liveAccumulators = accumulators;
    vectorStrategiesBySource = nextVectorStrategies;
    fusionStrategiesBySource = nextFusionStrategies;
    liveAccumulator = accumulators.old;
    detailCache.clear();
    if (!importedDetails) {
    adaptiveAccumulator = RetrievalMetrics.accumulator();
    const adaptiveRows = adaptiveManifest.items.map(row=>({...row}));
    const adaptiveByName = new Map(adaptiveRows.map(row=>[row.name,row]));
    const adaptiveFiles = Object.values(adaptiveManifest.detail_files);
    for (let offset=0; offset<adaptiveFiles.length; offset+=8) {
      const details = await Promise.all(adaptiveFiles.slice(offset,offset+8).map(readJSON));
      for (const detail of details) {
        if (detail.batch_id !== adaptiveManifest.batch_id) throw new Error('自适应基准批次不一致');
        const ranks = adaptiveAccumulator.add(detail);
        Object.assign(adaptiveByName.get(detail.name),ranks,{adaptive_rank:ranks.old_rank,query:detail.query});
      }
    }
    rowsByStrategy.adaptive = adaptiveRows;
    }
    refreshCalculatedTable();
    renderRouteFusionChart();
    refreshCasePanel();
    $('#metric-status').textContent = metricSource + ' · 已分别计算 ' + (strategies.length + (importedDetails ? 0 : 1)) + ' 组数据';
  } catch (error) {
    if (generation === metricGeneration) {
      liveAccumulator = null;
      liveAccumulators = {};
      $('#evaluation-panel').replaceChildren(el('div', 'error', '指标计算失败：' + error.message));
      $('#metric-status').textContent = '计算失败；未使用预存汇总值';
    }
  } finally {
    if (generation === metricGeneration) $('#recalculate-metrics').disabled = false;
  }
}
async function importEvaluation(file) {
  const details = RetrievalMetrics.evaluationDetails(await file.text());
  importedDetails = new Map(details.map(d => [d.name, d]));
  metricSource = '导入评测：' + file.name;
  rows = details.map(d => ({name:d.name,query:d.query,gold_count:d.gold.gold_chunk_ids.length}));
  selectedName = null;
  detailCache.clear();
  for (const id of ['strategy-select', 'case-strategy-select']) {
    const select = $('#' + id);
    for (const option of select.options) option.disabled = option.value !== 'old';
    select.value = 'old';
  }
  activeStrategy = caseStrategy = 'old';
  $('#filter').value = 'all';
  $('#search').value = '';
  await calculateLiveMetrics();
}


let batchPollInFlight = false;
async function checkLatestBatches() {
  if (document.hidden || importedDetails || !liveAccumulator || batchPollInFlight || $('#recalculate-metrics').disabled) return;
  batchPollInFlight = true;
  try {
    const current = await Promise.all(Object.values(DATASETS).map(async base => {
      try { return (await readJSON(base + 'index.json')).batch_id; }
      catch (error) { if (error.message.includes('HTTP 404')) return null; throw error; }
    }));
    const loaded = [adaptiveManifest?.batch_id || null, manifest?.batch_id || null, dimensionManifest?.batch_id || null, recommendedManifest?.batch_id || null];
    if (current.some((batch, i) => batch !== loaded[i])) {
      $('#metric-status').textContent = '发现新批次，正在重新读取并计算…';
      await reloadLatestMetrics();
    }
  } catch (error) {
    $('#metric-status').textContent = '自动检查最新批次失败：' + error.message;
  } finally { batchPollInFlight = false; }
}
setInterval(checkLatestBatches, 30000);
window.addEventListener('focus', checkLatestBatches);
document.addEventListener('visibilitychange', checkLatestBatches);
window.addEventListener('pipeline-strategies-changed', refreshCalculatedTable);
start();

function renderRecommended(detail, row, routes) {
  const experiment = detail.recommended;
  if (!experiment) return;
  const c = experiment.configuration, d = experiment.diagnostics;
  const diagnostics = el('details', 'dimension-info');
  diagnostics.append(el('summary', '', '推荐策略：原始 gap、维度角色与局部证据'));
  const gold = new Set(detail.gold?.gold_chunk_ids || []);
  for (const [key, p] of Object.entries(d.gaps)) {
    const status = !state(row).mapped ? 'Gold 未映射' : p.top1 ? gold.has(p.top1) ? 'Top-1 是 Golden' : 'Top-1 非 Golden' : '无候选';
    diagnostics.append(el('p', '', `${key === 'semantic' ? '语义' : '维度'}：gap ${p.gap.toFixed(4)}，相对 gap ${(p.relative_gap * 100).toFixed(2)}%，突出程度 ${p.prominence.toFixed(2)} 倍；${status}（仅评估）。`));
  }
  diagnostics.append(el('p', '', `同首位大 gap 保护：${d.shared_gap_active ? '触发' : '未触发'}；要求两路首位一致，且至少一路相对 gap ≥ ${(c.shared_gap_relative * 100).toFixed(1)}%、突出程度 ≥ ${c.shared_gap_prominence} 倍。大 gap 不代表候选正确。`),
    el('p', '', `当前维度${d.topic_only ? '仅表达泛化地点' : '包含具体内容或 POI'}；角色校准${d.role_calibration_active ? '启用' : '未启用'}。仅在泛化地点约束且语义相对 gap ≥ ${(c.role_semantic_gap * 100).toFixed(1)}% 时启用。实际维度缩放 ${d.dimension_scale.toFixed(4)}、词面缩放 ${d.lexical_scale.toFixed(4)}。`));
  for (const item of experiment.candidates) {
    const f = item.lexical_evidence;
    diagnostics.append(el('p', '', `#${item.rank} ${item.chunk_id}：局部窗口覆盖 ${f.window.toFixed(4)}；事实锚点 ${f.intent_anchor.toFixed(4)}（${f.matched_anchors.join('、') || '无命中'}）。`));
  }
  diagnostics.classList.add('strategy-diagnostics');
  routes.append(diagnostics);
}
