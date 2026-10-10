/* Read the current server configuration, independently of saved evaluation batches. */
(() => {
  let busy=false;
  const sources = {vector:['../../vector_retrieval/search.py','VECTOR_RETRIEVAL_STRATEGY'], fusion:['../fusion.py','FUSION_STRATEGY']};
  async function update() {
    if (busy || document.hidden) return;
    busy=true;
    const next={vector:null,fusion:null};
    const errors=[];
    await Promise.all(Object.entries(sources).map(async ([key,[path,constant]])=>{
      try {
        const url=new URL(path,location.href);url.searchParams.set('_fresh',Date.now());
        const response=await fetch(url,{cache:'no-store'});
        if (!response.ok) throw new Error('HTTP '+response.status);
        const source=await response.text();
        const match=source.match(new RegExp('^\\s*'+constant+'\\s*=\\s*["\']([^"\']+)["\']','m'));
        if (!match || !/^[A-Za-z_][A-Za-z0-9_]*\.py$/.test(match[1])) throw new Error('策略配置不是有效的文件名');
        next[key]=match[1];
      } catch(error) {errors.push(key+': '+error.message);}
    }));
    next.error=errors.join('；');
    currentPipelineStrategies=next;
    for (const node of document.querySelectorAll('[data-strategy-file]')) {
      const kind=node.closest('.semantic-intro,#semantic-comparison-panel') ? 'vector' : 'fusion';
      const active=node.dataset.strategyFile===next[kind];
      node.classList.toggle('is-mainline',active);
      node.querySelectorAll('.mainline-badge').forEach(badge=>badge.remove());
      if (active) (node.querySelector('h3,h2,th') || node).append(el('span','mainline-badge','当前主线'));
    }
    const options={adaptive:['自适应融合基准','online_adaptive.py'],new:['维度分数基准','dimension_score.py'],recommended:['维度角色校准','recommended_fusion.py']};
    for (const select of document.querySelectorAll('#strategy-select,#case-strategy-select')) {
      for (const option of select.options) {
        const method=options[option.value];
        if (method) option.textContent=method[0]+(method[1]===next.fusion ? ' · 当前主线' : '');
      }
    }
    window.dispatchEvent(new Event('pipeline-strategies-changed'));
    busy=false;
  }
  window.addEventListener('focus',update);
  document.addEventListener('visibilitychange',update);
  $('#recalculate-metrics').addEventListener('click',update);
  setInterval(update,30000);
  update();
})();
