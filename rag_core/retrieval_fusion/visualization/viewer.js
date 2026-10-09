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
const detailLoads = new Map();
const recommendedLoads = new Map();
let manifest = null;
let activeStrategy = 'recommended';
function offlineRank(row) { return activeStrategy === 'new' ? row.new_rank : row[`${activeStrategy}_rank`]; }
function offlineLabel() {
  return activeStrategy === 'recommended' ? window.RECOMMENDED_INDEX.label : '离线融合（维度分数基准）';
}
function isRank(rank, limit = 5) { return Number.isInteger(rank) && rank > 0 && rank <= limit; }
function state(row) {
  const mapped = Number.isInteger(row.gold_count) && row.gold_count > 0;
  return {
    mapped,
    semHit: mapped && isRank(row.semantic_rank),
    dimHit: mapped && isRank(row.dimension_rank),
    oldHit: mapped && isRank(row.old_rank),
    newHit: mapped && isRank(offlineRank(row)),
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
    metric('融合快照', total),
    metric('Gold 已映射', `${mapped} / ${total}`),
    metric('未映射', unmapped),
  );
}
function renderEvaluation(evaluation) {
  const panel = $('#evaluation-panel');
  const metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'MRR@10', 'nDCG@5'];
  if (!evaluation?.metrics?.length) {
    panel.hidden = true;
    return;
  }
  const head = el('div', 'evaluation-head');
  head.append(el('h2', '', '检索路线与融合策略对比'));
  const total = evaluation.total_queries ?? rows.length;
  head.append(el(
    'div',
    'evaluation-note',
    `指标均为百分制。Hit 分母为 ${evaluation.mapped_queries} 条 Gold 已映射查询；MRR@10 与 nDCG@5 按全部 ${total} 条快照计算。`,
  ));
  const makeTable = (rowsFor, mappedQueries, totalQueries) => {
    const table = el('table', 'evaluation-table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.append(el('th', '', '检索路线'));
    for (const label of metrics) headerRow.append(el('th', '', `${label} (%)`));
    thead.append(headerRow);
    const tbody = document.createElement('tbody');
    for (const row of rowsFor) {
      const tr = document.createElement('tr');
      if (row.key === 'fusion_new') tr.className = 'is-new';
      tr.append(el('th', '', row.label));
      for (const label of metrics) {
        const td = document.createElement('td');
        if (label.startsWith('Hit@')) {
          const count = row.hits?.[label] ?? (
            typeof row.scores?.[label] === 'number'
              ? Math.round(row.scores[label] * mappedQueries / 100)
              : null
          );
          td.textContent = Number.isInteger(count) && mappedQueries
            ? `${(count / mappedQueries * 100).toFixed(2)}% (${count}/${mappedQueries})`
            : '—';
        } else {
          const legacyPercent = row.scores?.[label];
          const value = row.ranking_scores?.[label] ?? (
            typeof legacyPercent === 'number'
              ? legacyPercent / 100 * mappedQueries / totalQueries
              : null
          );
          td.textContent = typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—';
        }
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(thead, tbody);
    const wrap = el('div', 'evaluation-table-wrap');
    wrap.append(table);
    return wrap;
  };
  panel.replaceChildren(head, makeTable([...evaluation.metrics, ...(window.RECOMMENDED_INDEX?.metrics || [])], evaluation.mapped_queries, total));
  panel.hidden = false;
}
function renderRouteFusionChart() {
  const panel = $('#route-fusion-panel');
  const categories = [
    { key: 'dimensionOnly', label: '仅维度检索 Top-5 命中' },
    { key: 'semanticOnly', label: '仅语义检索 Top-5 命中' },
    { key: 'bothRoutes', label: '语义与维度都命中' },
    { key: 'neitherRoute', label: '语义与维度都未命中' },
  ];
  const outcomes = [
    { key: 'bothFusion', label: '线上 + 离线都命中', short: '双融合命中', color: '#0b7a75' },
    { key: 'onlineOnly', label: '仅线上融合命中', short: '仅线上', color: '#4776a8' },
    { key: 'offlineOnly', label: '仅离线融合命中', short: '仅离线', color: '#d17a22' },
    { key: 'neitherFusion', label: '线上、离线都未命中', short: '双融合未中', color: '#89969d' },
  ];
  const buckets = Object.fromEntries(categories.map(category => [category.key, {
    total: 0, ...Object.fromEntries(outcomes.map(outcome => [outcome.key, 0])),
  }]));
  for (const row of rows) {
    const status = state(row);
    if (!status.mapped) continue;
    const category = status.dimHit
      ? (status.semHit ? 'bothRoutes' : 'dimensionOnly')
      : (status.semHit ? 'semanticOnly' : 'neitherRoute');
    const outcome = status.oldHit
      ? (status.newHit ? 'bothFusion' : 'onlineOnly')
      : (status.newHit ? 'offlineOnly' : 'neitherFusion');
    buckets[category].total += 1;
    buckets[category][outcome] += 1;
  }
  const mappedCount = rows.filter(row => state(row).mapped).length;
  const maxCategoryCount = Math.max(1, ...categories.map(category => buckets[category.key].total));
  const head = el('div', 'route-fusion-head');
  const title = el('h2', '', '两路 Top-5 命中组合与融合结果');
  title.id = 'route-fusion-title';
  head.append(title);
  head.append(el('div', 'route-fusion-note', `当前离线策略：${offlineLabel()}。仅统计 Gold 已映射的 ${mappedCount} 条查询；条块总长度按组内查询数缩放，颜色分段表示线上与当前离线策略的命中关系。`));
  const legend = el('div', 'route-fusion-legend');
  for (const outcome of outcomes) {
    const item = document.createElement('span');
    const dot = el('i', 'legend-dot');
    dot.style.backgroundColor = outcome.color;
    item.append(dot, document.createTextNode(outcome.label));
    legend.append(item);
  }
  const chartRows = el('div', 'route-fusion-rows');
  for (const category of categories) {
    const bucket = buckets[category.key];
    const line = el('div', 'route-fusion-row');
    const label = el('div', 'route-fusion-label');
    label.append(el('strong', '', category.label), el('span', '', `${bucket.total} 条`));
    const track = el('div', 'route-fusion-track');
    const bar = el('div', 'route-fusion-bar');
    bar.style.width = `${bucket.total / maxCategoryCount * 100}%`;
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', `${category.label}，共 ${bucket.total} 条`);
    for (const outcome of outcomes) {
      const count = bucket[outcome.key];
      if (!count) continue;
      const segment = el('div', 'route-fusion-segment', count);
      segment.style.flexGrow = String(count);
      segment.style.backgroundColor = outcome.color;
      segment.title = `${outcome.label}：${count} 条`;
      bar.append(segment);
    }
    track.append(bar);
    const counts = el('div', 'route-fusion-counts');
    for (const outcome of outcomes) {
      const count = el('div', 'route-fusion-count');
      count.title = outcome.label;
      count.append(el('strong', '', bucket[outcome.key]), document.createTextNode(outcome.short));
      counts.append(count);
    }
    line.append(label, track, counts);
    chartRows.append(line);
  }
  if (!mappedCount) chartRows.append(el('div', 'route-fusion-empty', '没有可统计的 Gold 映射数据。'));
  panel.replaceChildren(head, legend, chartRows);
}
function renderNotes() {
  const mapped = rows.filter(row => state(row).mapped);
  const oldLost = mapped.filter(row => { const s = state(row); return (s.semHit || s.dimHit) && !s.oldHit; }).length;
  const newLost = mapped.filter(row => { const s = state(row); return (s.semHit || s.dimHit) && !s.newHit; }).length;
  const rescued = mapped.filter(row => { const s = state(row); return !s.oldHit && s.newHit; }).length;
  const harmed = mapped.filter(row => { const s = state(row); return s.oldHit && !s.newHit; }).length;
  $('#strategy-notes').hidden = false;
  $('#badcase-summary').textContent = mapped.length
    ? `当前 ${rows.length} 条快照中，单路已进 Top-5、原融合却掉出的查询有 ${oldLost} 条；${offlineLabel()} 后还有 ${newLost} 条。相对线上融合，当前离线策略救回 ${rescued} 条、使 ${harmed} 条原 Top-5 命中掉出。`
    : '没有 Gold 排名；仍可逐条比较各路与离线融合结果。';
}
function passes(row, filter) {
  if (filter === 'all') return true;
  const s = state(row);
  if (filter === 'unmapped') return !s.mapped;
  if (!s.mapped) return false;
  if (filter === 'baseline-miss') return !s.oldHit;
  if (filter === 'new-fusion-miss') return !s.newHit;
  if (filter === 'rescued') return !s.oldHit && s.newHit;
  if (filter === 'harmed') return s.oldHit && !s.newHit;
  if (filter === 'offline-rescued') return !isRank(row.new_rank) && s.newHit;
  if (filter === 'offline-harmed') return isRank(row.new_rank) && !s.newHit;
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
    const label = !status.mapped ? '未映射 · 不计 Gold 指标' : `${offlineLabel()} ${status.newHit ? '命中' : '未命中'}`;
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
  for (const [name, key] of [['语义', 'semantic_rank'], ['维度', 'dimension_rank'], ['原融合', 'old_rank'], ['维度分数离线融合', 'new_rank']]) {
    const rank = row[key];
    const status = !state(row).mapped ? '未标注' : isRank(rank) ? `Top-5 命中 #${rank}` : Number.isInteger(rank) ? `第 ${rank} 名` : '未召回';
    badges.append(el('span', `badge${state(row).mapped ? isRank(rank) ? ' hit' : ' miss' : ''}`, `${name} ${status}`));
  }
  badges.append(el('span', 'badge gold', state(row).mapped ? `Golden ${row.gold_count} 个` : 'Gold 未映射'));
  if (activeStrategy !== 'new') {
    const rank = offlineRank(row);
    badges.append(el('span', `badge${state(row).mapped ? isRank(rank) ? ' hit' : ' miss' : ''}`,
      `${offlineLabel()} · ${state(row).mapped ? goldLabel(rank, true) : 'Gold 未映射'}`));
    if (state(row).mapped) badges.append(el('span', 'badge',
      `相对维度分数基准：${!isRank(row.new_rank) && isRank(rank) ? '救回 Top-5' : isRank(row.new_rank) && !isRank(rank) ? '掉出 Top-5' : 'Top-5 命中状态相同'}`));
  }
  const routes = el('div', 'routes');
  routes.append(
    renderRoute(detail, row, 'semantic', '语义检索', '相似度排序', 'semantic_rank'),
    renderRoute(detail, row, 'dimension', '维度检索', '维度匹配排序', 'dimension_rank'),
    renderRoute(detail, row, 'old', '原融合', '来自 output', 'old_rank'),
    renderRoute(detail, row, 'new', '离线融合（维度分数）', '保留基准 · output_new', 'new_rank'),
  );
  renderRecommended(detail, row, routes);
  const foot = el('div', 'detail-foot');
  foot.append(el('span', 'chunk-hint', '点击 chunk 编号查看快照中保存的正文。'));
  if (state(row).mapped) foot.append(el('span', '', 'Gold 标记显示各路首个 golden；维度分数融合在离线快照上计算，线上策略保持不变。'));
  root.replaceChildren(head, badges, renderConstraints(detail), routes, foot);
}
async function loadDetail() {
  const root = $('#detail');
  const row = filtered.find(item => item.name === selectedName);
  if (!row) { root.replaceChildren(el('div', 'empty', '没有符合条件的问题')); return; }
  const name = row.name;
  if (detailCache.has(name) && detailCache.get(name).recommended) { renderDetail(detailCache.get(name), row); return; }
  root.replaceChildren(el('div', 'empty', '正在读取本地静态详情…'));
  try {
    let detail = detailCache.get(name) || window.RETRIEVAL_FUSION_DETAILS?.[name];
    if (!detail) {
      let load = detailLoads.get(name);
      if (!load) {
        const path = manifest?.detail_files?.[name];
        if (!path) throw new Error('静态索引中没有这条问题的详情文件');
        load = new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = `./${path}`;
          script.onload = () => {
            script.remove();
            const details = window.RETRIEVAL_FUSION_DETAILS;
            const loaded = details?.[name];
            if (loaded) {
              delete details[name];
              resolve(loaded);
            } else reject(new Error('详情文件已加载，但没有找到对应数据'));
          };
          script.onerror = () => {
            script.remove();
            reject(new Error(`无法读取静态详情文件：${path}`));
          };
          document.head.append(script);
        });
        detailLoads.set(name, load);
      }
      try {
        detail = await load;
        detailLoads.delete(name);
      } catch (error) {
        detailLoads.delete(name);
        throw error;
      }
    }
    const recommendedPath = window.RECOMMENDED_INDEX?.detail_files?.[name];
    if (recommendedPath && !detail.recommended) {
      let load = recommendedLoads.get(name);
      if (!load) load = new Promise((resolve, reject) => {
        const script = document.createElement('script'); script.src = `./${recommendedPath}`;
        script.onload = () => {
          script.remove(); const payload = window.RECOMMENDED_DETAILS?.[name];
          if (!payload) { reject(new Error('Recommended detail missing')); return; }
          delete window.RECOMMENDED_DETAILS[name]; resolve(payload);
        };
        script.onerror = () => { script.remove(); reject(new Error(`Cannot read ${recommendedPath}`)); };
        document.head.append(script);
      });
      recommendedLoads.set(name, load);
      try { detail.recommended = await load; } finally { recommendedLoads.delete(name); }
    }
    if (detailCache.size >= 8) detailCache.delete(detailCache.keys().next().value);
    detailCache.set(name, detail);
    if (selectedName === name) renderDetail(detail, row);
  } catch (error) {
    if (selectedName === name) root.replaceChildren(el('div', 'error', `读取快照失败：${error.message}`));
  }
}
async function start() {
  try {
    manifest = window.RETRIEVAL_FUSION_INDEX;
    if (!Array.isArray(manifest?.items) || !manifest.items.length) {
      throw new Error('静态索引为空；请重新运行 build_static_data.py 生成页面数据包');
    }
    rows = manifest.items.map(row => ({...row, ...(window.RECOMMENDED_INDEX?.items?.[row.name] || {})}));
    setupRecommended();
    $('#subtitle').textContent = '比较语义检索、维度检索、线上融合、维度分数基准和推荐融合；支持逐题诊断。';
    renderMetrics(manifest.evaluation);
    renderEvaluation(manifest.evaluation);
    renderRouteFusionChart();
    renderNotes();
    const hasGold = rows.some(row => state(row).mapped);
    const unmappedCount = rows.filter(row => !state(row).mapped).length;
    const unmappedOption = $('#filter').querySelector('option[value="unmapped"]');
    if (unmappedOption) unmappedOption.textContent = `未映射（${unmappedCount}）`;
    for (const option of $('#filter').options) {
      if (option.value !== 'all') option.disabled = !hasGold && option.value !== 'unmapped';
    }
    $('#search').addEventListener('input', applyFilters);
    $('#filter').addEventListener('change', applyFilters);
    $('#show-all').addEventListener('click', () => {
      showAll = !showAll;
      $('#show-all').textContent = showAll ? '仅显示 Top-5' : '展开全部候选';
      loadDetail();
    });
    $('#chunk-dialog-close').addEventListener('click', () => $('#chunk-dialog').close());
    applyFilters();
  } catch (error) {
    $('#app').replaceChildren(el('div', 'error', `未能加载融合快照：${error.message}`));
  }
}
start();

function renderRecommended(detail, row, routes) {
  const experiment = detail.recommended;
  if (!experiment) return;
  routes.append(renderRoute({...detail, routes: {...detail.routes, recommended: experiment.candidates}},
    row, 'recommended', window.RECOMMENDED_INDEX.label, '局部证据 · 维度角色校准', 'recommended_rank'));
  const c = experiment.configuration, d = experiment.diagnostics;
  const diagnostics = el('details', 'dimension-info'); diagnostics.open = true;
  diagnostics.append(el('summary', '', '推荐策略：原始 gap、维度角色与局部证据'));
  const gold = new Set(detail.gold?.gold_chunk_ids || []);
  for (const [key, p] of Object.entries(d.gaps)) {
    const status = !state(row).mapped ? 'Gold 未映射' : p.top1 ? gold.has(p.top1) ? 'Top-1 是 Golden' : 'Top-1 非 Golden' : '无候选';
    diagnostics.append(el('p', '', `${key === 'semantic' ? '语义' : '维度'}：gap ${p.gap.toFixed(4)}，相对 gap ${(p.relative_gap * 100).toFixed(2)}%，突出程度 ${p.prominence.toFixed(2)} 倍；${status}（仅评估）。`));
  }
  diagnostics.append(el('p', '', `同首位大 gap 保护：${d.shared_gap_active ? '触发' : '未触发'}；要求两路首位一致，且至少一路相对 gap ≥ ${(c.shared_gap_relative * 100).toFixed(1)}%、突出程度 ≥ ${c.shared_gap_prominence} 倍。大 gap 不代表候选正确。`),
    el('p', '', `当前维度${d.topic_only ? '仅表达泛化地点' : '包含具体内容或 POI'}；角色校准${d.role_calibration_active ? '启用' : '未启用'}。仅在泛化地点约束且语义相对 gap ≥ ${(c.role_semantic_gap * 100).toFixed(1)}% 时启用。实际维度缩放 ${d.dimension_scale.toFixed(4)}、词面缩放 ${d.lexical_scale.toFixed(4)}。`));
  for (const item of experiment.candidates.slice(0, 5)) {
    const f = item.lexical_evidence;
    diagnostics.append(el('p', '', `#${item.rank} ${item.chunk_id}：局部窗口覆盖 ${f.window.toFixed(4)}；事实锚点 ${f.intent_anchor.toFixed(4)}（${f.matched_anchors.join('、') || '无命中'}）。`));
  }
  routes.append(diagnostics);
}

function setupRecommended() {
  const data = window.RECOMMENDED_INDEX;
  if (!data) return;
  const panel = $('#recommended-summary'), r = data.report, m = r.metrics;
  panel.hidden = false;
  panel.append(el('h2', '', data.label),
    el('p', '', `Top-5 ${(m.hit_at_5 / m.mapped_queries * 100).toFixed(2)}%（${m.hit_at_5}/${m.mapped_queries}）；相对维度分数基准救回 ${r.changes_vs_baseline.rescued} 条、掉出 ${r.changes_vs_baseline.harmed} 条。`),
    el('p', '', `剩余 ${r.remaining_badcases} 条 Top-5 badcase：${r.candidate_pool_missing} 条 Golden 未进入候选池，${r.remaining_badcases - r.candidate_pool_missing} 条已进入但排名靠后。使用“当前离线 Top-5 未命中”筛选查看。`),
    el('p', '', '推荐融合结合两路得分与名次、320 字局部证据和查询事实锚点；维度只表达泛化地点时有条件地调整贡献。Gold 仅用于评估。指标为当前历史快照上的结果。'));
  $('#strategy-select').addEventListener('change', () => {
    activeStrategy = $('#strategy-select').value;
    renderRouteFusionChart(); renderNotes(); applyFilters();
  });
}

