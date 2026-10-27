"""Mechanical JSON escaping only; decoded scores and reasons stay unchanged."""
import hashlib
import json
import re


def canonicalize(original,expected_fields):
    try:json.loads(original);return original,None
    except json.JSONDecodeError:pass
    try:loose=json.loads(original,strict=False)
    except json.JSONDecodeError:loose=None
    if (isinstance(loose,dict) and set(loose)==set(expected_fields)
            and all(isinstance(v,dict) and set(v)=={'score','reasoning'} and isinstance(v['reasoning'],str) for v in loose.values())
            and any(any(ord(char)<32 for char in v['reasoning']) for v in loose.values())):
        # JSON's strict=False merely reads literal control characters. Re-encode
        # them with JSON escapes, preserving every decoded reason character.
        canonical=json.dumps(loose,ensure_ascii=False,indent=2)
        assert json.loads(canonical)==loose
        return canonical,{'kind':'escape_literal_reason_control_characters',
            'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
            'decoded_values_unchanged':True,'no_score_or_word_edits':True}
    # A Chinese closing quotation mark is content, not JSON's ASCII delimiter.
    # Insert only the missing delimiter after an otherwise completed reason.
    closers=[]
    for match in re.finditer(r'(?m)^[ \t]*"reasoning"\s*:\s*"([^\r\n]*?)([ \t]*)(?=\r?\n[ \t]*\})',original):
        if match.group(1).endswith('”'):closers.append({'offset':match.end(1),'insert':'"'})
    if closers:
        candidate=original
        for edit in reversed(closers):candidate=candidate[:edit['offset']]+'"'+candidate[edit['offset']:]
        try:closed=json.loads(candidate)
        except json.JSONDecodeError:closed=None
        if (isinstance(closed,dict) and set(closed)==set(expected_fields)
                and all(isinstance(v,dict) and set(v)=={'score','reasoning'} and isinstance(v['reasoning'],str) for v in closed.values())):
            return candidate,{'kind':'insert_missing_reason_ascii_closer_only',
                'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                'canonical_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
                'edits':closers,'no_score_or_word_edits':True}
    matches=list(re.finditer(r'(?m)^[ \t]*"reasoning"\s*:\s*"([^\r\n]*)"(?=[ \t]*\r?\n[ \t]*\})',original))
    if len(matches)!=len(expected_fields):raise ValueError('Not an unambiguous completed reason-line quote error')
    edits=[]
    for match in matches:
        body=match.group(1)
        for index,char in enumerate(body):
            if char!='"':continue
            backslashes=0;position=index-1
            while position>=0 and body[position]=='\\':backslashes+=1;position-=1
            if backslashes%2==0:edits.append({'offset':match.start(1)+index,'insert':'\\'})
    if not edits:raise ValueError('No safe quote-only repair')
    canonical=original
    for edit in sorted(edits,key=lambda value:value['offset'],reverse=True):canonical=canonical[:edit['offset']]+'\\'+canonical[edit['offset']:]
    value=json.loads(canonical)
    if not isinstance(value,dict) or set(value)!=set(expected_fields):raise ValueError('Quote repair changed object scope')
    return canonical,{'kind':'escape_only_unescaped_interior_reason_quotes','original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'edits':edits,'no_score_or_word_edits':True}
