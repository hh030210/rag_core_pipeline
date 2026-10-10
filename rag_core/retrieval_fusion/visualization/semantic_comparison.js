/* Read experiment candidates only; never generate or persist metric summaries. */
(() => {
  const base = '../../vector_retrieval/experiments/00_method_comparison/output/dataset/';
  const labels = {
    subqueries_concat:'子查询拼接', original_only:'仅原始查询',
    original_plus_subqueries_concat:'原始查询加子查询拼接',
    original_subqueries_centroid_w2:'查询向量加权平均（原始权重 2）',
    dense_ensemble:'纯向量集成'
  };
  let snapshot = null, busy = false;
  const percent = value => (value * 100).toFixed(2) + '%';
  function render() {
    if (!snapshot) return;
    const {index, sums, total, mapped} = snapshot;
    const denominator = $('#metric-denominator').value === 'all' ? total : mapped;
    const config = index.configuration || {};
    const host = $('#semantic-comparison-content');
    const note = el('p','evaluation-note',
      '数据来源：vector_retrieval/experiments/00_method_comparison/output/dataset · 保存查询 ' + total
      + ' · 当前主线：' + (currentPipelineStrategies.vector || '配置读取失败或读取中')
      + ' · Gold 已映射 ' + mapped + ' · 当前分母 ' + denominator
      + ' · 原始输入 ' + (config.input_count ?? total) + '，实验已排除未映射查询 ' + (config.excluded_unmapped_count ?? total - mapped)
      + '。得分由候选排序与 Golden 现算；平均子检索次数按每条查询的逻辑向量搜索数统计。');
    host.replaceChildren(note);
    const wrap = el('div','evaluation-table-wrap');
    const table = el('table','evaluation-table semantic-table');
    const caption = el('caption','sr-only','全部语义检索策略的效果与检索成本对比');
    table.append(caption);
    const head = el('thead'), hr = el('tr');
    for (const label of ['策略','平均子检索次数','Hit@1 (%)','Hit@5 (%)','Hit@10 (%)','Hit@15 (%)','MRR@10 (%)','nDCG@5 (%)']) {
      const th = el('th','',label); th.scope='col'; hr.append(th);
    }
    head.append(hr);table.append(head);
    const body = el('tbody');
    for (const method of index.methods) {
      const s = sums[method], row = el('tr'); row.dataset.method=method;row.dataset.strategyFile=method+'.py';
      const th = el('th','',labels[method] || method); th.scope='row';
      th.append(el('code','',method + '.py'));
      if (method+'.py' === currentPipelineStrategies.vector) {th.append(el('span','mainline-badge','当前主线'));row.classList.add('is-mainline');}
      row.append(th);
      row.append(el('td','',s.costCount===total ? (s.cost / total).toFixed(2) : '未完整记录'));
      for (const k of [1,5,10,15]) row.append(el('td','',denominator ? percent(s.hits['Hit@'+k]/denominator)+' ('+s.hits['Hit@'+k]+'/'+denominator+')' : '—'));
      for (const metric of ['mrr','ndcg']) row.append(el('td','',denominator ? percent(s[metric]/denominator) : '—'));
      body.append(row);
    }
    table.append(body);wrap.append(table);host.append(wrap);
  }
  async function load(index = null) {
    if (busy) return;
    busy=true;
    try {
      index ||= await readJSON(base + 'index.json');
      if (!Array.isArray(index.methods) || !index.methods.length) throw new Error('方法清单为空');
      const files = Object.values(index.detail_files || {});
      if (!files.length) throw new Error('候选详情为空');
      const sums = Object.fromEntries(index.methods.map(m=>[m,{hits:Object.fromEntries([1,5,10,15].map(k=>['Hit@'+k,0])),mrr:0,ndcg:0,cost:0,costCount:0}]));
      let mapped=0;
      for (let offset=0; offset<files.length; offset+=8) {
        const details=await Promise.all(files.slice(offset,offset+8).map(file=>readJSON(base+file)));
        for (const detail of details) {
          if (detail.batch_id !== index.batch_id) throw new Error('实验正在更新，批次不一致，请稍后重试');
          if (!Array.isArray(detail.gold_chunk_ids)) throw new Error('缺少 Golden 标注');
          if (detail.gold_chunk_ids.length) mapped++;
          for (const method of index.methods) {
            const results=detail.strategies?.[method]?.results;
            if (!Array.isArray(results)) throw new Error(method+' 缺少候选排序');
            const metric=RetrievalMetrics.score(results,detail.gold_chunk_ids), s=sums[method];
            for (const k in s.hits) s.hits[k]+=metric.hits[k];
            s.mrr+=metric.mrr;s.ndcg+=metric.ndcg;
            const cost=detail.cost?.[method]?.logical_dense_searches;
            if (Number.isFinite(cost)) {s.cost+=cost;s.costCount++;}
          }
        }
      }
      snapshot={index,sums,total:files.length,mapped};render();
    } catch(error) {
      snapshot=null;
      $('#semantic-comparison-content').replaceChildren(el('div','error','语义策略对比加载失败：'+error.message));
    } finally {busy=false;}
  }
  async function check() {
    if (document.hidden || busy) return;
    try {
      const index=await readJSON(base+'index.json');
      if (!snapshot || JSON.stringify(index)!==JSON.stringify(snapshot.index)) await load(index);
    } catch(error) {
      snapshot=null;
      $('#semantic-comparison-content').replaceChildren(el('div','error','语义实验检查失败：'+error.message));
    }
  }
  window.addEventListener('pipeline-strategies-changed',render);
  $('#metric-denominator').addEventListener('change',render);
  $('#recalculate-metrics').addEventListener('click',()=>load());
  setInterval(check,30000);
  window.addEventListener('focus',check);
  document.addEventListener('visibilitychange',check);
  load();
})();
