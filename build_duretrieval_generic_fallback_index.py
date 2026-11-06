"""Build a separate DuRetrieval dimension index with generic fallback facets."""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from code_jyx.index_builder import IndexBuilderV2

EXAM = re.compile(r"(?:选择题|填空题|阅读理解|试卷|答案|解析|第\s*\d+\s*[题小]|A[\.．、]|B[\.．、])", re.I)
FILL = re.compile(r"(?:填空|_{3,}|\(\s*\))")
CHOICE = re.compile(r"(?:A[\.．、].{0,50}B[\.．、]|选择题)", re.I)
QA = re.compile(r"(?:简答|阅读|回答问题|问答)")
LIST = re.compile(r"(?:^|\n)\s*(?:\d+|[一二三四五六七八九十]+)[、.．)]", re.M)

NEW_NODES = [
    ("generic_text_profile", "通用文本属性", None, 1, False, "通用语料的文本形态与学科主题。"),
    ("text_genre", "文本类型", "generic_text_profile", 2, True, "试题、答案解析、诗文、知识陈述等文本类型。"),
    ("subject_topic", "学科主题", "generic_text_profile", 2, True, "文本所属的粗粒度学科或主题。"),
    ("question_format", "题型", "generic_text_profile", 2, True, "选择、填空、阅读问答等题目形式。"),
    ("content_structure", "内容形态", "generic_text_profile", 2, True, "选项列表、编号列表、连续段落等结构。"),
]

def add_label(doc, dim, label):
    doc.setdefault("tags", {}).setdefault(dim, [])
    if label not in doc["tags"][dim]: doc["tags"][dim].append(label)
    doc.setdefault("tag_details", {}).setdefault(dim, [])
    if not any(x.get("normalized_label") == label for x in doc["tag_details"][dim]):
        doc["tag_details"][dim].append({"label": label, "raw_label": label, "normalized_label": label, "evidence": "deterministic_generic_fallback", "confidence": 1.0})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-run',required=True); p.add_argument('--output-run',required=True); a=p.parse_args()
    src, out=Path(a.source_run),Path(a.output_run); out.mkdir(parents=True,exist_ok=True)
    schema=json.loads((src/'V_core_v2.json').read_text(encoding='utf-8'))
    schema['dimensions'] += [{"id":i,"name":n,"parent_id":par,"level":lev,"indexable":idx,"description":desc,"aliases":[],"value_policy":"multi","max_labels":5} for i,n,par,lev,idx,desc in NEW_NODES]
    tags=json.loads((src/'tags_output_v2.json').read_text(encoding='utf-8')); chunks={x['doc_id']:x for x in json.loads((src/'chunks.json').read_text(encoding='utf-8'))}
    for doc_id,doc in tags['documents'].items():
        text=str(chunks[doc_id].get('chunk_text_full',''))
        add_label(doc,'text_genre','试题/答案材料' if EXAM.search(text) else '通用知识文本')
        add_label(doc,'question_format','选择题' if CHOICE.search(text) else ('填空题' if FILL.search(text) else ('阅读/问答' if QA.search(text) else '非题目文本')))
        add_label(doc,'content_structure','列表/选项' if LIST.search(text) or CHOICE.search(text) else '连续段落')
        add_label(doc,'subject_topic','语文/人文' if re.search(r'诗|文言|阅读|作文|词语',text) else '通用')
    (out/'V_core_v2.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'V_cand_v2.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'tags_output_v2.json').write_text(json.dumps(tags,ensure_ascii=False),encoding='utf-8')
    index=IndexBuilderV2(schema,output_dir=out).build(tags); IndexBuilderV2(schema,output_dir=out).save(index,out)
    shutil.copy2(src/'chunks.json',out/'chunks.json')
    print(json.dumps({'output':str(out),'documents':len(tags['documents']),'new_leaf_dimensions':4},ensure_ascii=False))
if __name__=='__main__': main()
