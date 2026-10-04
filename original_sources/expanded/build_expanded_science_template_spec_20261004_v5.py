"""Format manuscript Markdown as a supplied-template document specification.

This is a presentation-only parser. It neither computes scientific results nor
certifies editorial review. Wide tables are partitioned by columns, retaining
every source cell and repeated identifying columns. All partitions are recorded.
"""
from pathlib import Path
import argparse, hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]


def bind(p):
    b = p.read_bytes()
    return {'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def caption(s, kind='Table'):
    s = re.sub(r'\s+', ' ', s).strip()
    m = re.match(r'^' + kind + r' (S?\d+)\. (.+?\.)(.*)$', s)
    assert m, s
    return '**' + kind + ' ' + m[1] + '. ' + m[2] + '**' + m[3]


def widths(n, rows):
    if n == 2: return [4680, 4680]
    if n == 3: return [3360, 3000, 3000]
    if n == 4:
        if rows[0][0] == 'Study': return [1800,2160,2520,2880]
        if rows[0][0] == 'Domain': return [1200,3960,1800,2400]
        if rows[0][1] == 'Endpoint content': return [1800,2160,2880,2520]
        return [3360, 2000, 2000, 2000]
    if n == 5: return [2760, 1800, 1600, 1600, 1600]
    if n == 6: return [2760, 1560, 1260, 1260, 1260, 1260]
    if n == 7: return [2400, 1020, 1020, 1020, 1300, 1300, 1300]
    raise ValueError('Partition tables wider than seven columns')


def table_parts(rows):
    n = len(rows[0]); assert all(len(row) == n for row in rows)
    # Source columns remain literal: no abbreviation or rounding of stored values.
    if n <= 6: return [list(range(n))]
    # A split table must repeat all row identifiers, not just a duplicated head
    # label. Seed and endpoint distinguish measurements on otherwise equal heads.
    if rows[0][:2] in (['Head', 'Seed'], ['Head', 'Endpoint']):
        identifiers = [0, 1, 2] if rows[0][2] == 'Endpoint' else [0, 1]
        return [[*identifiers, *range(start, min(start+3, n))]
                for start in range(len(identifiers), n, 3)]
    if rows[0][:2] in (['Endpoint', 'Contrast'], ['Endpoint', 'Aligned minus']):
        return [[0, 1, *range(start, min(start+3, n))] for start in range(2, n, 3)]
    if n == 7 and 'Precision %' in rows[0]: return [[0, 1, 2, 3], [0, 4, 5, 6]]
    if n == 7 and rows[0][0] == 'Domain': return [[0, 1, 2, 3], [0, 1, 4, 5, 6]]
    if n == 10 and rows[0][:2] == ['Head', 'Seed']: return [[0, 1, 2, 3], [0, 1, 4, 5, 6], [0, 1, 7, 8, 9]]
    if n == 9 and rows[0][0] == 'Head': return [[0, 1, 2, 3], [0, 4, 5, 6, 7, 8]]
    if n > 7:
        return [[0, *range(start, min(start + 4, n))] for start in range(1, n, 4)]
    return [list(range(n))]


def build(source, destination, allow_pending):
    assert not destination.exists(), 'Preserve previous document specifications'
    text = source.read_text(); lines = text.splitlines()
    assert allow_pending or '{{' not in text, 'Scientific result placeholders remain'
    title = lines[0].removeprefix('# ').strip()
    sections = [{'supplement': False, 'blocks': [{'type': 'title', 'text': title}]},
                {'supplement': True, 'blocks': []}]
    section = sections[0]; paragraphs = []; pending_caption = None
    table_records = []; abstract_next = False; i = 1; in_refs = False

    def flush():
        nonlocal paragraphs, abstract_next
        if paragraphs:
            prose = re.sub(r'\s+', ' ', ' '.join(paragraphs)).strip()
            prose = prose.replace('`', '')
            b = {'type': 'paragraph', 'text': prose}
            if abstract_next: b['bold'] = True; abstract_next = False
            section['blocks'].append(b); paragraphs = []

    while i < len(lines):
        s = lines[i].strip()
        if not s: flush(); i += 1; continue
        if s.startswith('Authors, '):
            flush(); section['blocks'].append({'type': 'authors', 'text': s}); i += 1; continue
        if s.startswith('# Supplementary Materials'):
            flush(); section = sections[1]; in_refs = False; i += 1; continue
        if s.startswith('#'):
            flush(); heading = s.lstrip('#').strip()
            if heading == 'Abstract': abstract_next = True; i += 1; continue
            in_refs = heading == 'References and Notes'
            section['blocks'].append({'type': 'heading', 'text': heading, 'level': 2 if s.startswith('###') else 1})
            i += 1; continue
        if in_refs and (m := re.match(r'^(\d+)\. (.+)$', s)):
            flush(); section['blocks'].append({'type': 'reference', 'number': int(m[1]), 'text': m[2]}); i += 1; continue
        if re.match(r'^Table S?\d+\.', s):
            flush(); cap_lines = [s]; i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith('|'):
                cap_lines.append(lines[i].strip()); i += 1
            pending_caption = caption(' '.join(cap_lines)); continue
        if s.startswith('|'):
            flush(); assert pending_caption is not None, 'Each table needs a source caption'
            rows = []; original_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                line = lines[i].strip(); original_lines.append(line)
                row = [c.strip() for c in line.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c.replace(' ', '')) for c in row): rows.append(row)
                i += 1
            partitions = table_parts(rows)
            assert set(j for part in partitions for j in part) == set(range(len(rows[0])))
            for number, columns in enumerate(partitions, 1):
                selected = [[row[j] for j in columns] for row in rows]
                cp = pending_caption + (f' Column panel {number} of {len(partitions)}; identifying columns repeat.' if len(partitions) > 1 else '')
                section['blocks'].append({'type': 'caption', 'text': cp})
                section['blocks'].append({'type': 'table', 'widths': widths(len(columns), selected), 'rows': selected,
                    'rowsPerPage': 5 if any(len(c) > 100 for row in selected for c in row) else 8})
            table_records.append({'caption': pending_caption, 'source_rows': rows, 'source_table_lines_sha256': hashlib.sha256('\n'.join(original_lines).encode()).hexdigest(),
                'column_partitions': partitions, 'every_source_column_retained': True})
            # S3 has two explicitly named model subtables under one caption.
            if not pending_caption.startswith('**Table S3.'): pending_caption = None
            continue
        if s.startswith('!['):
            flush(); m = re.match(r'^!\[(.*?)\]\((.*?)\)$', s); assert m, s
            p = Path(m[2]); p = p if p.is_absolute() else ROOT / p
            assert p.is_file(); assert i + 2 < len(lines)
            j = i + 1
            while not lines[j].strip(): j += 1
            cap = lines[j].strip(); assert cap.startswith('**Figure ')
            section['blocks'].extend([{'type': 'figure', 'path': str(p.relative_to(ROOT)), 'alt': m[1], 'widthFraction': 1, 'pageBreakBefore': True},
                                     {'type': 'caption', 'text': cap}]); i = j + 1; continue
        paragraphs.append(s); i += 1
    flush()
    refs = [b for section in sections for b in section['blocks'] if b['type'] == 'reference']
    assert len(refs) == 26 and sorted(b['number'] for b in refs) == list(range(1, 27))
    assert any(b.get('bold') for b in sections[0]['blocks']), 'Actual abstract required'
    spec = {'schema': 'science_template_word_companion_v1', 'title': title,
            'status': 'Working formatting proof with pending evidence' if allow_pending else 'Research manuscript; model-assisted editorial records accompany the saved draft',
            'sections': sections}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    record = {'scope': 'Markdown-to-template presentation mapping; no score or editorial certificate',
              'manuscript': {'path': str(source.relative_to(ROOT)), **bind(source)},
              'specification': {'path': str(destination.relative_to(ROOT)), **bind(destination)},
              'parser': {'path': str(Path(__file__).resolve().relative_to(ROOT)), **bind(Path(__file__))},
              'table_partitions': table_records, 'scientific_values_computed': False}
    destination.with_suffix('.table_mapping.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'tables': len(table_records), 'references': len(refs), 'specification': record['specification']}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('source', type=Path); ap.add_argument('destination', type=Path)
    ap.add_argument('--allow-working-pending', action='store_true'); a = ap.parse_args()
    build(a.source.resolve(), a.destination.resolve(), a.allow_working_pending)
