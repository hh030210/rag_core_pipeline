"""Narrow structural recovery; preserve original scores and all reason text."""
import hashlib
import json
import re

from knowledge_view_format import canonicalize as original_canonicalize


def canonicalize(original,view,fields):
    candidate=original;repairs=[]
    try:value=json.loads(candidate)
    except json.JSONDecodeError:
        # Only one exact missing delimiter: reason followed by named continuation.
        pattern=r'(?m)(^[ \t]*"reasoning"\s*:\s*"(?:[^"\\\r\n]|\\.)*")([ \t]*\r?\n[ \t]*"reasoning_additional"\s*:)'
        candidate,n=re.subn(pattern,r'\1,\2',candidate)
        if n:repairs.append({'kind':'insert_missing_comma_before_named_reason_continuation','count':n})
        # Remove only a trailing delimiter after a completed named reason.
        pattern=r'(?m)(^[ \t]*"reasoning"\s*:\s*"(?:[^"\\\r\n]|\\.)*")[ \t]*,([ \t]*\r?\n[ \t]*\})'
        candidate,n=re.subn(pattern,r'\1\2',candidate)
        if n:repairs.append({'kind':'remove_trailing_comma_after_completed_reason','count':n})
        try:value=json.loads(candidate)
        except json.JSONDecodeError:
            # Only append an object closer after already closed metric objects;
            # a truncated/unclosed reason string remains invalid and is rejected.
            if not candidate.rstrip().endswith('}'):return original_canonicalize(original,fields)
            try:value=json.loads(candidate+'}')
            except json.JSONDecodeError:return original_canonicalize(original,fields)
            candidate+='}';repairs.append({'kind':'append_one_missing_root_object_closer'})
    if isinstance(value,dict) and set(value)=={view} and isinstance(value[view],dict) and set(value[view])==set(fields):
        value=value[view];repairs.append({'kind':'unwrap_exact_matching_view_envelope','view':view})
    elif isinstance(value,dict) and set(value)=={'evaluation_view','metrics'} and value['evaluation_view']==view and isinstance(value['metrics'],dict) and set(value['metrics'])==set(fields):
        value=value['metrics'];repairs.append({'kind':'unwrap_exact_view_identity_and_metrics_envelope','view':view})
    if not isinstance(value,dict) or set(value)!=set(fields):return original_canonicalize(original,fields)
    before=json.loads(json.dumps(value,ensure_ascii=False));merged=[]
    for key,item in value.items():
        if not isinstance(item,dict):return original_canonicalize(original,fields)
        if set(item)=={'score','reasoning','reasoning_additional'}:
            if not all(isinstance(item[k],str) and item[k].strip() for k in ('reasoning','reasoning_additional')):return original_canonicalize(original,fields)
            value[key]={'score':item['score'],'reasoning':item['reasoning']+'\n'+item['reasoning_additional']}
            merged.append(key)
        elif set(item)!={'score','reasoning'}:return original_canonicalize(original,fields)
    if merged:repairs.append({'kind':'join_explicit_reason_continuation_without_word_edits','metrics':merged,'separator':'\n'})
    if not repairs:return original_canonicalize(original,fields)
    for key,item in value.items():
        assert item['score']==before[key]['score']
        expected=before[key]['reasoning']+('\n'+before[key]['reasoning_additional'] if key in merged else '')
        assert item['reasoning']==expected
    canonical=json.dumps(value,ensure_ascii=False,indent=2)
    return canonical,{'kind':'named_structural_recovery_only','repairs':repairs,
        'original_normalized_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
        'original_metric_objects':before,'scores_unchanged':True,'all_reason_text_preserved':True}
