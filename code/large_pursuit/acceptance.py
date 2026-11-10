"""Formal dispatch requires a reproducible repair receipt and fresh pilot evidence."""
import hashlib
import json
from pathlib import Path
from .provenance import ROOT,sources,sha
from .semantics import SCHEMA


def acceptance_sources():
    result=sources()
    for relative in ('code/tests/test_large_pursuit.py','code/tests/test_large_pursuit_semantics.py',
                     'code/tests/fixtures/large_pursuit_rare_negative.json',
                     'code/verify_large_pursuit.py','code/verify_large_pursuit_repair.py'):
        path=ROOT/relative
        if path.exists():result[relative]=sha(path)
    return result


def require_repair(out):
    out=Path(out);path=out/'repair_acceptance.json'
    if (out/'audit_hold.json').exists():raise RuntimeError('Formal experiment blocked by unresolved code audit')
    if not path.exists():raise RuntimeError('Missing repair acceptance receipt')
    receipt=json.loads(path.read_text())
    if receipt.get('schema')!=SCHEMA or receipt.get('status')!='passed' or receipt.get('sources')!=acceptance_sources():
        raise RuntimeError('Repair receipt is stale or incomplete')
    if set(receipt.get('closed_blockers',[]))!={'B%02d'%i for i in range(1,11)}:raise RuntimeError('Unclosed plan-compliance blockers')
    return receipt


def require_pilot(out):
    require_repair(out);path=Path(out)/'pilot_audit.json'
    if not path.exists():raise RuntimeError('Fresh pilot audit required before formal budget freeze')
    audit=json.loads(path.read_text())
    if audit.get('status')!='fresh_pilot_reviewed' or audit.get('sources')!=acceptance_sources():
        raise RuntimeError('Fresh pilot audit is stale/incomplete')
    return audit
