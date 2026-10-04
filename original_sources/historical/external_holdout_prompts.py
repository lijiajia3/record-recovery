"""Exact source-derived frozen prompt builders for separately locked holdout.

Generation only uses source and released task inputs, with the same train-only
retrieval demonstrations as development. No target-file loader lives here.
The two blocks are source-derived once from the original run_stage branches and
checked against captured original prompt strings before any holdout request.
"""
import json

def polyie_prompt(task, inp, stage, retriever, get):
    old = None
    p = task.extraction_prompt(inp, retriever.get(inp) if retriever else None)
    if stage in ('baseline', 'retrieval'):
        pass
    elif stage in ('generic', 'targeted'):
        old = get('retrieval')
        p += '\nCURRENT COMPLETE GROUPS:\n' + task.prompt_json(task.groups_for_prompt(old['relations']))
        if stage == 'generic':
            p += '\nReview carefully for incorrect groups and omissions. Return a COMPLETE corrected list with exact evidence.'
        else:
            p += '\nAudit each complete material/property/value/condition binding against the source. Check whether conditions qualify this exact measurement or a different experiment. Check all provided materials and values, range endpoints, negations and contrasts. Preserve correct groups; remove unsupported bindings; recover explicitly missing groups. Return a COMPLETE corrected list with exact evidence.'
    elif stage in ('generic_twice', 'source_confirmation'):
        old = get('generic' if stage == 'generic_twice' else 'targeted')
        p += '\nCURRENT COMPLETE GROUPS:\n' + task.prompt_json(task.groups_for_prompt(old['relations']))
        if stage == 'generic_twice':
            p += '\nReview carefully again for incorrect groups and omissions. Return a COMPLETE corrected list with exact evidence.'
        else:
            p += '\nPerform final source confirmation of every complete group. Check each role and the scope of every condition, distinguish separate experiments, verify quotes and recover omissions. Return a COMPLETE corrected list with exact evidence.'
    elif stage == 'verify_edits':
        old, proposed = (get('retrieval'), get('targeted'))
        edits = task.edits_for(inp, old, proposed)
        if not edits:
            return (None, old, {'relations': [], 'no_proposed_edits': True})
        p = task.TASK_SCHEMA + '\nJudge candidate edits to scientific measurement groups using ONLY the source and GIVEN entities. Evaluate the entire CN/PN/PV/Condition binding, including absence of conditions. A quote containing all names is necessary but does not by itself prove their binding. For BOTH operations the judgment describes factual support of the supplied GROUP, not whether you approve the edit. A supported addition can be accepted; a supported old group must be preserved. For a deletion, judge the OLD full binding; delete only when the source establishes wrong association or a contradicting binding, not just uncertainty. Use supported, unsupported, or uncertain. Give an ordinal confidence integer 1 to 5: 1 guess, 2 weak, 3 plausible, 4 strong explicit evidence, 5 unambiguous explicit evidence. Return compact JSON with a relations list. Each judgment object has edit_id, judgment (supported, unsupported or uncertain), confidence (integer 1 through 5), and evidence (a string of actual copied SOURCE words covering every group endpoint occurrence). Never put instructions or placeholder text in evidence. Return each edit ID exactly once. Do not modify candidate groups or invent new ones.\nSOURCE:\n' + task.prompt_json({'text': inp['text'], 'entities': task.prompt_entities(inp)}) + '\nTRAINING EXAMPLES FROM OTHER PAPERS:\n' + task.prompt_json(retriever.get(inp)) + '\nCANDIDATE EDITS:\n' + task.prompt_json([dict(e, group=task.groups_for_prompt([e['group']])[0]) for e in edits])
    else:
        raise ValueError(stage)
    return (p, old, None)


def mulms_prompt(task, inp, stage, retriever, get):
    p = task.prompt(inp, retriever.get(inp) if retriever else None)
    old = None
    if stage in ['baseline', 'retrieval']:
        pass
    elif stage in ['generic', 'targeted', 'generic_twice', 'source_confirmation']:
        parent = {'generic': 'retrieval', 'targeted': 'retrieval', 'generic_twice': 'generic', 'source_confirmation': 'targeted'}[stage]
        old = get(parent)
        p += '\nCURRENT COMPLETE EXTRACTION:\n' + json.dumps(old['relations'], ensure_ascii=False)
        if stage in ['generic', 'generic_twice']:
            p += '\nReview carefully for wrong edges and missing relations. Return a COMPLETE corrected list with exact evidence.'
        else:
            p += '\nAudit source support, head/tail direction, exact entity occurrence and type, measurement result versus condition, separate experiments, ranges and negations. Preserve correct edges and recover explicit omissions. Return a COMPLETE corrected list with exact evidence.'
    elif stage == 'verify_edits':
        old = get('retrieval')
        proposed = get('targeted')
        edits = task.edits_for(inp, old, proposed)
        if not edits:
            return (None, old, {'relations': [], 'no_proposed_edits': True})
        p = task.TASK_SCHEMA + '\nJudge proposed edits to directed scientific relations using ONLY SOURCE. Check the exact head/tail occurrences, entity types, direction and label semantics, including condition versus measured result. A quote covering names does not prove binding. For BOTH operations the judgment is factual support of the supplied EDGE, not whether you approve the edit. Judge deletions by support of the OLD edge. Delete only when the source establishes an unsupported association, not uncertainty. Return one compact JSON object with a relations list. Every judgment has edit_id, judgment (supported, unsupported or uncertain), confidence (integer 1 through 5), and evidence (a string of actual copied SOURCE words, never a placeholder). Confidence 1 guess,2 weak,3 plausible,4 strong explicit evidence,5 unambiguous explicit evidence. Return each edit ID once, with quote covering the actual endpoint occurrences. Never invent another candidate.\nSOURCE:\n' + inp['text'] + '\nTRAINING EXAMPLES FROM OTHER PAPERS:\n' + json.dumps(retriever.get(inp), ensure_ascii=False) + '\nEDITS:\n' + json.dumps(edits, ensure_ascii=False)
    else:
        raise ValueError(stage)
    return (p, old, None)
