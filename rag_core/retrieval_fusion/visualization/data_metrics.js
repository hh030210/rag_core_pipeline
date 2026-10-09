(function(root) {
  'use strict';
  const routes = {semantic:'语义检索', dimension:'维度检索', old:'线上融合',
                  new:'离线融合（维度分数基准）', recommended:'推荐融合'};
  function ids(items) {
    return [...new Set(items.map(x => String(x.chunk_id ?? x.id ?? '')).filter(Boolean))];
  }
  function score(items, goldValues) {
    const gold = new Set(goldValues.map(String)), ordered = ids(items);
    const first = ordered.findIndex(id => gold.has(id));
    const rank = first < 0 ? null : first + 1;
    const ideal = Array.from({length: Math.min(5, gold.size)}, (_,i) => 1/Math.log2(i+2))
                       .reduce((a,b)=>a+b,0);
    const dcg = ordered.slice(0,5).reduce((sum,id,i) =>
      sum + (gold.has(id) ? 1/Math.log2(i+2) : 0), 0);
    return {rank, hits: Object.fromEntries([1,5,10,15].map(k=>['Hit@'+k, Number(rank!==null && rank<=k)])),
            mrr: rank!==null && rank<=10 ? 1/rank : 0, ndcg: ideal ? dcg/ideal : 0};
  }
  function accumulator() {
    let total=0, mapped=0;
    const sums={};
    return {
      add(detail) {
        const gold=detail.gold?.gold_chunk_ids;
        if (!Array.isArray(gold)) throw new Error('详情缺少 gold.gold_chunk_ids，无法计算指标');
        total++; if (gold.length) mapped++;
        const rankFields={semantic:'semantic_rank',dimension:'dimension_rank',
                          old:'old_rank',new:'new_rank',recommended:'recommended_rank'};
        const ranks={gold_count:new Set(gold.map(String)).size};
        const lists={...detail.routes};
        if (detail.recommended) lists.recommended=detail.recommended.candidates;
        for (const [route,label] of Object.entries(routes)) {
          if (!Array.isArray(lists[route])) continue;
          const m=score(lists[route],gold);
          const s=sums[route] ||= {key:route,label,available:0,mapped:0,
              hits:{'Hit@1':0,'Hit@5':0,'Hit@10':0,'Hit@15':0},mrr:0,ndcg:0};
          s.available++; if (gold.length) s.mapped++;
          for (const k in s.hits) s.hits[k]+=m.hits[k];
          s.mrr+=m.mrr; s.ndcg+=m.ndcg;
          ranks[rankFields[route]]=m.rank;
        }
        return ranks;
      },
      result(mode='mapped') {
        const denominator=mode==='all' ? total : mapped;
        return {total_queries:total,mapped_queries:mapped,unmapped_queries:total-mapped,
          denominator,denominator_mode:mode,metrics:Object.values(sums).map(s=>({
            key:s.key,label:s.label,hits:s.hits,available:s.available,
            complete:s.available===total,
            ranking_scores:{'MRR@10':denominator ? s.mrr/denominator : 0,
                            'nDCG@5':denominator ? s.ndcg/denominator : 0}
          }))};
      }
    };
  }
  function evaluationDetails(text) {
    let records;
    const trimmed=text.trim();
    if (!trimmed) throw new Error('评测文件为空');
    if (trimmed.startsWith('[')) records=JSON.parse(trimmed);
    else records=trimmed.split(/\r?\n/).filter(x=>x.trim()).map((line,i)=>{
      try {return JSON.parse(line);} catch(e) {throw new Error('第 '+(i+1)+' 行 JSON 无效');}
    });
    if (!Array.isArray(records) || !records.length) throw new Error('评测文件没有记录');
    return records.map((r,i)=>{
      const gold=r.gold?.chunk_ids ?? r.gold_chunk_ids;
      const retrieval=r.retrieval;
      if (!Array.isArray(gold) || !retrieval) throw new Error('第 '+(i+1)+' 条不是 evaluate 结果');
      const get=(name)=>{
        const list=retrieval[name+'_candidates'] ?? retrieval[name+'_results'];
        if (!Array.isArray(list)) throw new Error('第 '+(i+1)+' 条缺少 '+name+' 候选');
        return list.map((h,j)=>({...h,rank:j+1}));
      };
      const lists={semantic:get('semantic'),dimension:get('dimension'),old:get('fusion')};
      const chunks={};
      for (const list of Object.values(lists)) for (const h of list) {
        const id=String(h.chunk_id); chunks[id] ||= {
          title:h.doc_title || h.chunk_gen_title || '',source_file:h.source_file || '',
          text:h.chunk_text_full || h.chunk_text || h.text || ''};
      }
      return {name:'evaluate_'+String(i).padStart(4,'0'),query:r.question || r.query || '',
        query_analysis:r.query_analysis || {},gold:{gold_chunk_ids:gold.map(String)},
        routes:lists,chunks};
    });
  }
  const api={score,accumulator,evaluationDetails};
  if (typeof module!=='undefined' && module.exports) module.exports=api;
  root.RetrievalMetrics=api;
})(typeof window!=='undefined' ? window : globalThis);

