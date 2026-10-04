"""Prospective source-text provenance audit. No annotation/model/network access.

Real execution requires an explicitly authorized, hash-bound source-only gate.
All JSON Gold values in already exposed POLYIE sources are lexically skipped;
MuLMS Parquet is projected to doc_id/sentence only, in bounded batches.
Similarity is descriptive, never a split-changing or independence certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import unicodedata

TEXT_ARCHIVE = 'SC-CoMIcs-Abstract1000_CC_BY-NC-3.0.zip'
ANN_ARCHIVE = 'SC-CoMIcs-Annotation1000_CC_BY_4.0.zip'
NATIVE_IDS = set(range(1, 1001))
NATIVE_SPLITS = {'train': list(range(201, 1001)), 'dev': list(range(101, 201)), 'test': list(range(1, 101))}
LEGACY_PATHS = {
    **{f'data/polyie/source/{s}.json': ('POLYIE', s, 'json_tokens') for s in NATIVE_SPLITS},
    **{f'data/mulms/source/{s}.parquet': ('MuLMS', s, 'parquet_sentences') for s in NATIVE_SPLITS},
}
ALLOWED_META_FIELDS = {'id', 'doc_id', 'name', 'doi', 'DOI', 'title', 'year', 'publication_year', 'article_type', 'type', 'set'}
HEX = re.compile(r'^[0-9a-f]{64}$')
WORD = re.compile(r'\w+', re.UNICODE)


class AuditError(ValueError):
    def __init__(self, code, detail):
        super().__init__(f'{code}: {detail}')
        self.code, self.detail = code, detail


def require(condition, code, detail):
    if not condition:
        raise AuditError(code, detail)


def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def normalize_doi(value):
    if value is None:
        return ''
    require(isinstance(value, str), 'invalid_doi_type', type(value).__name__)
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value.strip(), flags=re.I).casefold()


def valid_doi(value):
    return bool(re.fullmatch(r'10\.\d{4,9}/\S+', value))


def split_for(source_id):
    return 'test' if source_id <= 100 else 'dev' if source_id <= 200 else 'train'


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate_json_key', key)
        result[key] = value
    return result


class SourceAccess:
    """Only declared regular source/document files, never symlink resolution."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        require('..' not in self.root.parts, 'unsafe_root', str(root))
        for part in (self.root, *self.root.parents):
            require(not part.is_symlink(), 'symlink_path', str(part))
        require(self.root.is_dir(), 'missing_root', str(self.root))
        self.root = self.root.resolve()
        self.open_log = []

    def checked(self, path, *, exists=True):
        supplied = Path(path)
        require('..' not in supplied.parts, 'unsafe_path', str(path))
        target = supplied if supplied.is_absolute() else self.root / supplied
        require(target.is_relative_to(self.root), 'outside_root', str(path))
        require(not any(p.lower() in {'raw_annotations', 'checkpoints', 'cache', 'caches'} for p in target.parts),
                'forbidden_namespace', str(path))
        require(target.suffix.casefold() not in {'.ann', '.xmi', '.pt', '.pth', '.bin', '.pkl', '.pickle', '.safetensors', '.zip'},
                'forbidden_extension', str(path))
        for part in (target, *target.parents):
            require(not part.is_symlink(), 'symlink_path', str(part))
        if exists:
            require(target.exists() and stat.S_ISREG(target.lstat().st_mode), 'missing_regular_file', str(target))
        return target

    def open(self, path, kind):
        target = self.checked(path)
        fd = os.open(target, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'nonregular_open', str(target))
        self.open_log.append({'path': str(target.relative_to(self.root)), 'kind': kind})
        return os.fdopen(fd, 'rb')

    def hash(self, path, kind='raw_byte_integrity_only'):
        h, size = hashlib.sha256(), 0
        with self.open(path, kind) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block); size += len(block)
        return h.hexdigest(), size

    def verified(self, path, wanted, size=None):
        require(isinstance(wanted, str) and HEX.fullmatch(wanted), 'invalid_hash_binding', str(path))
        sha, actual_size = self.hash(path)
        require(sha == wanted and (size is None or size == actual_size), 'source_hash_or_size_mismatch',
                {'path': str(path), 'actual_sha256': sha, 'actual_bytes': actual_size})
        return {'path': str(self.checked(path).relative_to(self.root)), 'sha256': sha, 'bytes': actual_size}

    def json(self, path, kind):
        with self.open(path, kind) as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        require(len(raw) <= 16 * 1024 * 1024, 'oversize_metadata_json', str(path))
        return json.loads(raw, object_pairs_hook=unique_object), digest_bytes(raw)


class ProjectedArray:
    """Streaming top-level object projection; omitted values are not decoded.

    Unknown strings/containers are consumed lexically without json.loads,
    object creation, field traversal, label counting or annotation evaluation.
    Only one selected record is materialized, not a full JSON corpus/cache.
    """
    def __init__(self, stream, fields):
        import io
        self.stream = io.TextIOWrapper(stream, encoding='utf-8', newline='')
        self.fields, self.buf, self.pos = set(fields), '', 0

    def peek(self):
        if self.pos == len(self.buf):
            self.buf, self.pos = self.stream.read(8192), 0
        return self.buf[self.pos] if self.buf else ''

    def get(self):
        c = self.peek()
        if c: self.pos += 1
        return c

    def ws(self):
        while self.peek() and self.peek().isspace(): self.get()

    def take(self, expected):
        self.ws(); got = self.get()
        require(got == expected, 'invalid_projection_json_structure', {'expected': expected, 'got': got})

    def string(self, capture):
        require(self.get() == '"', 'invalid_json_string', 'missing quote')
        out = ['"'] if capture else None
        while True:
            c = self.get()
            require(bool(c), 'invalid_json_string', 'unterminated')
            if capture: out.append(c)
            if capture: require(len(out) <= 8 * 1024 * 1024, 'oversize_selected_field', '8MiB per string')
            if c == '\\':
                escaped = self.get(); require(bool(escaped), 'invalid_json_string', 'unterminated escape')
                if capture: out.append(escaped)
            elif c == '"': break
        return ''.join(out) if capture else None

    def value(self, capture):
        self.ws(); first = self.peek()
        require(bool(first), 'invalid_projection_json_structure', 'missing value')
        if first == '"': return self.string(capture)
        out = [] if capture else None
        captured_size = 0
        if first in '[{':
            stack = []
            while True:
                c = self.peek(); require(bool(c), 'invalid_projection_json_structure', 'unterminated container')
                if c == '"':
                    s = self.string(capture)
                    if capture:
                        out.append(s); captured_size += len(s)
                        require(captured_size <= 8 * 1024 * 1024, 'oversize_selected_field', '8MiB per field')
                    continue
                c = self.get()
                if capture:
                    out.append(c); captured_size += 1
                if c in '[{': stack.append(']' if c == '[' else '}')
                elif c in ']}':
                    require(bool(stack) and stack.pop() == c, 'invalid_projection_json_structure', 'mismatched bracket')
                    if not stack: break
                if capture: require(captured_size <= 8 * 1024 * 1024, 'oversize_selected_field', '8MiB per field')
        else:
            n = 0
            while self.peek() and self.peek() not in ',]} \t\r\n':
                c = self.get(); n += 1
                if capture: out.append(c)
            require(n > 0, 'invalid_projection_json_structure', 'empty scalar')
        return ''.join(out) if capture else None

    def rows(self):
        self.take('['); self.ws()
        if self.peek() == ']': self.get(); return
        while True:
            self.take('{'); row, seen = {}, set(); self.ws()
            if self.peek() != '}':
                while True:
                    self.ws(); key = json.loads(self.string(True))
                    require(key not in seen, 'duplicate_json_key', key); seen.add(key)
                    self.take(':'); selected = key in self.fields
                    raw = self.value(selected)
                    if selected: row[key] = json.loads(raw)
                    self.ws(); c = self.get()
                    require(c in ',}', 'invalid_projection_json_structure', c)
                    if c == '}': break
            else: self.get()
            yield row
            self.ws(); c = self.get()
            require(c in ',]', 'invalid_projection_json_structure', c)
            if c == ']': break
        self.ws(); require(not self.peek(), 'trailing_projection_json', 'unexpected data')


def project_json(access, path, fields):
    with access.open(path, 'json_source_field_projection_no_gold_decode') as stream:
        yield from ProjectedArray(stream, fields).rows()


def parquet_rows(access, path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AuditError('parquet_dependency_unavailable', 'No install or network attempted') from exc
    with access.open(path, 'parquet_projection_doc_id_sentence_only') as stream:
        file = pq.ParquetFile(stream)
        for batch in file.iter_batches(batch_size=256, columns=['doc_id', 'sentence']):
            yield from batch.to_pylist()


def private_source_snapshot(access, original, binding, destination, number):
    """Stream-copy an already exposed legacy container, verify exact bytes.

    The raw container may encode old Gold; its values are never decoded here.
    The projection reads this private copy, not a replaceable original path.
    Copies are removed on successful commit and preserved only on failure.
    """
    folder=destination/'__source_projections__';folder.mkdir(exist_ok=True)
    path=folder/f'source_{number:04d}{Path(original).suffix}'
    h,size=hashlib.sha256(),0
    with access.open(original,'old_source_raw_byte_snapshot_no_semantic_parse') as source, path.open('xb') as output:
        for block in iter(lambda:source.read(1024*1024),b''):
            output.write(block);h.update(block);size+=len(block)
    require(h.hexdigest()==binding['sha256'] and size==binding['bytes'],'legacy_snapshot_hash_mismatch',str(original))
    access.verified(path,binding['sha256'],binding['bytes'])
    return {'path':str(path.relative_to(access.root)),'sha256':binding['sha256'],'bytes':binding['bytes']}


def metadata_rows(access, spec):
    require(Path(spec['path']).suffix.casefold() in {'.json', '.csv'}, 'invalid_metadata_file_extension', str(spec['path']))
    binding = access.verified(spec['path'], spec['sha256'], spec.get('bytes'))
    fields = set(spec['fields'])
    require(fields <= ALLOWED_META_FIELDS, 'forbidden_metadata_field', sorted(fields - ALLOWED_META_FIELDS))
    if spec['format'] == 'json_array':
        yield binding, project_json(access, spec['path'], fields)
    elif spec['format'] == 'csv':
        import io
        with access.open(spec['path'], 'official_source_metadata_csv_only') as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
            yield binding, ({k: row.get(k) for k in fields} for row in reader)
    else: raise AuditError('unsupported_metadata_format', spec['format'])


def load_gate(access, gate_path, manifest_path, expected_sha, synthetic):
    gate, gate_sha = access.json(gate_path, 'source_only_protocol_gate')
    wanted_status = 'synthetic_fixture_only' if synthetic else 'authorized_real_source_text_only_audit'
    require(gate.get('status') == wanted_status, 'protocol_not_authorized', gate.get('status'))
    require(gate.get('annotation_semantics_allowed') is False and gate.get('model_work_allowed') is False,
            'not_source_only_gate', 'annotation/model permissions must be false')
    require(gate.get('fixed_similarity') == {'word_ngram': 5, 'jaccard_numerator': 4, 'jaccard_denominator': 5, 'min_word_tokens': 20},
            'changed_similarity_contract', gate.get('fixed_similarity'))
    own_sha = digest_bytes(Path(__file__).read_bytes())
    require(gate.get('script_sha256') == own_sha, 'unreviewed_script_hash', own_sha)
    require(gate.get('acquisition_manifest_sha256') == expected_sha, 'manifest_gate_binding_mismatch', expected_sha)
    if not synthetic:
        require(gate.get('root_confirmed_real_acquisition') is True and gate.get('actual_execution_authorized') is True,
                'real_acquisition_not_confirmed', 'Root confirmation and explicit real-execution authorization required')
    require(Path(manifest_path).name == 'source_acquisition_manifest.json', 'unexpected_manifest_name', str(manifest_path))
    manifest, manifest_sha = access.json(manifest_path, 'confirmed_acquisition_metadata_only')
    require(manifest_sha == expected_sha, 'acquisition_manifest_hash_mismatch', manifest_sha)
    require(manifest.get('protocol_sha256') == gate.get('source_stage_protocol_sha256'), 'source_stage_binding_mismatch', manifest.get('protocol_sha256'))
    for flag in ['semantic_annotation_read', 'label_statistics_computed', 'model_training_or_inference_performed']:
        require(manifest.get(flag) is False, 'acquisition_scope_violation', flag)
    require(manifest.get('paid_api_calls') == 0 and manifest.get('all_saved_bytes_rehashed_after_all_copies') is True,
            'acquisition_integrity_or_scope_missing', 'Required completion metadata absent')
    require(manifest.get('split_ids') == NATIVE_SPLITS, 'changed_native_split', 'Expected800/100/100 native fold1')
    require(manifest.get('native_source_abstract_records') == 1000, 'wrong_native_population', 'Expected1000 source IDs')
    require(set(manifest['official_archives']) == {TEXT_ARCHIVE, ANN_ARCHIVE}, 'changed_archive_registry', 'Expected pinned archive names')
    require(set(manifest['normalized_doi_by_id']) == {str(i) for i in NATIVE_IDS}, 'incomplete_native_doi_ids', 'Expected IDs1–1000')
    require(manifest['official_archives'][TEXT_ARCHIVE].get('annotation_semantics_parsed') is False,
            'archive_scope_violation', 'Source archive must be semantically unparsed')
    return gate, gate_sha, manifest, manifest_sha, own_sha


def schema(db):
    db.executescript('''
    CREATE TABLE units(uid INTEGER PRIMARY KEY, corpus TEXT, split TEXT, source_id TEXT, document_id TEXT,
      doi TEXT, repr_kind TEXT, source_sha TEXT, byte_sha TEXT, projected_sha TEXT, norm_sha TEXT,
      norm_text TEXT, word_tokens INTEGER, gram_count INTEGER);
    CREATE TABLE grams(gram TEXT, uid INTEGER REFERENCES units(uid));
    CREATE INDEX gram_index ON grams(gram,uid);
    CREATE TABLE pairs(kind TEXT, left_uid INTEGER, right_uid INTEGER, intersection_count INTEGER,
      union_count INTEGER, similarity REAL);
    ''')


def add_unit(db, *, corpus, split, source_id, document_id, doi, text, repr_kind, source_sha, byte_sha=None):
    require(isinstance(text, str), 'invalid_source_text_type', str(source_id))
    norm = normalized_text(text); tokens = WORD.findall(norm)
    grams = {' '.join(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))}
    row = (corpus, split, str(source_id), str(document_id), doi, repr_kind, source_sha, byte_sha,
           digest_bytes(text.encode('utf-8')), digest_bytes(norm.encode('utf-8')), None, len(tokens), len(grams))
    cursor = db.execute('INSERT INTO units(corpus,split,source_id,document_id,doi,repr_kind,source_sha,byte_sha,projected_sha,norm_sha,norm_text,word_tokens,gram_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', row)
    uid = cursor.lastrowid
    if len(tokens) >= 20:
        db.executemany('INSERT INTO grams VALUES(?,?)', ((g, uid) for g in sorted(grams)))
    return uid


def eligible(left, right, *, near=False):
    # A study-source comparison always has at least one native SC-CoMIcs unit.
    if 'SC-CoMIcs' not in (left[1], right[1]): return False
    return left[1] != right[1] or left[2] != right[2] or left[2] == right[2] == 'test' or not near


def duplicate_pairs(db):
    for kind, column in [('exact_native_source_bytes', 'byte_sha'), ('exact_projected_text_utf8', 'projected_sha'), ('normalized_text', 'norm_sha'), ('normalized_doi', 'doi')]:
        query = f'SELECT {column} FROM units WHERE {column} IS NOT NULL AND {column} != ? GROUP BY {column} HAVING COUNT(*)>1'
        for (value,) in db.execute(query, ('',)):
            rows = list(db.execute(f'SELECT uid,corpus,split FROM units WHERE {column}=? ORDER BY uid', (value,)))
            for left, right in itertools.combinations(rows, 2):
                if eligible(left, right): db.execute('INSERT INTO pairs VALUES(?,?,?,NULL,NULL,1.0)', (kind,left[0],right[0]))


def near_pairs(db):
    # Exhaustive inverted-index intersection: every J>=0.8 pair shares a gram.
    # Integer comparison avoids floating-point behavior at exactly4/5.
    db.execute('''INSERT INTO pairs
      SELECT 'word5gram_jaccard_ge_0.8', a.uid,b.uid,COUNT(*),a.gram_count+b.gram_count-COUNT(*),
             1.0*COUNT(*)/(a.gram_count+b.gram_count-COUNT(*))
      FROM grams g JOIN grams h ON g.gram=h.gram AND g.uid<h.uid
      JOIN units a ON a.uid=g.uid JOIN units b ON b.uid=h.uid
      WHERE (a.corpus='SC-CoMIcs' OR b.corpus='SC-CoMIcs')
        AND (a.corpus<>b.corpus OR a.split<>b.split OR (a.split='test' AND b.split='test'))
        AND a.word_tokens>=20 AND b.word_tokens>=20
      GROUP BY a.uid,b.uid
      HAVING 5*COUNT(*)>=4*(a.gram_count+b.gram_count-COUNT(*))''')


def native_test_clusters(db):
    """Fixed100-node test-induced text graph, including every isolated root.

    Connectivity uses source-text exact/normalized/near edges, not Gold or
    external-corpus bridges. This supplies prospective resampling groups only.
    """
    ids=NATIVE_SPLITS['test'];parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    edges=set()
    for left,right in db.execute('''SELECT a.source_id,b.source_id FROM pairs p
        JOIN units a ON a.uid=p.left_uid JOIN units b ON b.uid=p.right_uid
        WHERE a.corpus='SC-CoMIcs' AND b.corpus='SC-CoMIcs' AND a.split='test' AND b.split='test'
        AND p.kind IN ('exact_native_source_bytes','normalized_text','word5gram_jaccard_ge_0.8')'''):
        left,right=sorted((int(left),int(right)));edges.add((left,right))
        a,b=find(left),find(right)
        if a!=b:parent[max(a,b)]=min(a,b)
    components={}
    for i in ids:components.setdefault(find(i),[]).append(i)
    groups=[]
    for first,members in sorted(components.items()):
        dois=[r[0] for i in members for r in db.execute("SELECT doi FROM units WHERE corpus='SC-CoMIcs' AND source_id=?",(str(i),))]
        groups.append({'cluster_id':f'sccomics_test_text_cluster_{first:04d}','native_test_ids':members,'normalized_dois':dois,
          'native_record_count':len(members),'distinct_text_edges':sum(a in members and b in members for a,b in edges),
          'independent_experiments_certified':False})
    return {'scope':'fixed100 native test DOI-record nodes; induced within-test source-text graph only',
      'edge_views':['exact_native_source_bytes','normalized_text','word5gram_jaccard_ge_0.8'],
      'threshold':{'word_ngram':5,'jaccard_numerator':4,'jaccard_denominator':5,'min_word_tokens':20},
      'native_test_record_count':100,'source_text_cluster_count':len(groups),'distinct_within_test_text_edges':len(edges),
      'clusters':groups,'all_test_ids_present_exactly_once':sorted(i for g in groups for i in g['native_test_ids'])==ids,
      'gold_or_model_used':False,'external_or_crossfold_bridges_used':False,'native_split_changed':False,
      'resampling_or_statistical_test_performed':False,'same_experiment_or_pretraining_independence_certified':False}


def _run_audit(root, acquisition_manifest, expected_manifest_sha, source_only_protocol, output_dir, *, synthetic=False, attempt=None):
    access = SourceAccess(root)
    gate, gate_sha, manifest, manifest_sha, own_sha = load_gate(access, source_only_protocol, acquisition_manifest, expected_manifest_sha, synthetic)
    destination = access.checked(output_dir, exists=False)
    require(not destination.exists(), 'preserve_existing_output', str(destination))
    require(destination.is_relative_to(access.root / 'research'), 'output_outside_research', str(destination))
    members = manifest['official_archives'][TEXT_ARCHIVE]['members']; by_id = {}
    for original_name, entry in members.items():
        if entry.get('category') != 'raw_source_document': continue
        name = PurePosixPath(original_name)
        require(not name.is_absolute() and '..' not in name.parts and '\\' not in original_name and '\x00' not in original_name,
                'unsafe_native_member_name', original_name)
        require(name.suffix == '.txt' and re.fullmatch(r'\d+', name.stem), 'invalid_native_text_member', original_name)
        source_id = int(name.stem)
        require(source_id in NATIVE_IDS and source_id not in by_id, 'duplicate_or_unexpected_source_id', source_id)
        path = access.checked(entry['canonical_path'])
        require(path.name == f'{source_id:04d}.txt' and path.parent == access.root / 'data/sccomics_round4/source_v3/raw_text',
                'unexpected_native_canonical_text_path', str(path))
        by_id[source_id] = entry
    require(set(by_id) == NATIVE_IDS, 'incomplete_native_text_ids', {'missing': sorted(NATIVE_IDS-set(by_id))})
    destination.mkdir(parents=True)
    if attempt is not None:attempt['created_output']=destination
    db = sqlite3.connect(destination / 'source_text_index.sqlite3'); schema(db)
    bindings, licenses, missing_dois, empty_texts, empty_legacy_texts,working_copies = [], [], [], [], [], []
    metadata_by_doi, legacy_meta = {}, {}
    for spec in gate.get('official_metadata_inputs', []):
        for binding, rows in metadata_rows(access, spec):
            bindings.append(binding)
            for row in rows:
                doi = normalize_doi(row.get('doi', row.get('DOI', '')))
                if valid_doi(doi):
                    # Preserve repeated official metadata, never replace silently.
                    metadata_by_doi.setdefault(doi, []).append(row)
                corpus = spec.get('corpus')
                doc_id = row.get('doc_id', row.get('name', row.get('id')))
                if corpus and doc_id is not None:
                    legacy_meta.setdefault((corpus,str(doc_id)), []).append(row)
    for spec in gate.get('license_inputs', []):
        binding = access.verified(spec['path'], spec['sha256'], spec.get('bytes'))
        licenses.append({**binding, 'declared_scope': spec['declared_scope'], 'license_name': spec.get('license_name'), 'source_url': spec.get('source_url')})
    for source_id in sorted(NATIVE_IDS):
        entry = by_id[source_id]; binding = access.verified(entry['canonical_path'], entry['sha256'], entry['bytes'])
        bindings.append(binding)
        with access.open(entry['canonical_path'], 'native_source_text_only') as stream: raw = stream.read()
        require(digest_bytes(raw) == entry['sha256'], 'source_changed_during_read', source_id)
        text = raw.decode('utf-8'); doi = normalize_doi(manifest['normalized_doi_by_id'][str(source_id)])
        if not valid_doi(doi): missing_dois.append({'corpus':'SC-CoMIcs','document_id':str(source_id),'invalid_or_missing_doi':doi})
        if not text.strip(): empty_texts.append(source_id)
        add_unit(db,corpus='SC-CoMIcs',split=split_for(source_id),source_id=source_id,document_id=source_id,doi=doi if valid_doi(doi) else '',
                 text=text,repr_kind='native_original_utf8_text_bytes',source_sha=binding['sha256'],byte_sha=binding['sha256'])
    for number,spec in enumerate(gate.get('legacy_source_inputs', [])):
        target = access.checked(spec['path']); rel = str(target.relative_to(access.root))
        require(rel in LEGACY_PATHS, 'unapproved_legacy_source_path', rel)
        require(tuple(spec[k] for k in ['corpus','split','format']) == LEGACY_PATHS[rel], 'legacy_source_identity_mismatch', rel)
        binding = access.verified(target,spec['sha256'],spec.get('bytes')); bindings.append(binding)
        copied=private_source_snapshot(access,target,binding,destination,number);working_copies.append(copied)
        projected_path=access.root/copied['path']
        rows = project_json(access,projected_path,{'id','text'}) if spec['format']=='json_tokens' else parquet_rows(access,projected_path)
        seen = set()
        for n,row in enumerate(rows):
            if spec['format']=='json_tokens':
                doc_id = str(row['id']); require(doc_id not in seen,'duplicate_legacy_document_id', {'path':rel,'id':doc_id});seen.add(doc_id)
                require(isinstance(row['text'],list) and all(isinstance(t,str) for t in row['text']), 'invalid_poly_text_projection', doc_id)
                text = ' '.join(row['text']); source_id = doc_id; repr_kind='released_token_list_space_join_not_original_article_bytes'
            else:
                doc_id = str(row['doc_id']); text = row['sentence']; source_id=f'{doc_id}:row{n}';repr_kind='released_sentence_field_not_original_article_bytes'
            metas = legacy_meta.get((spec['corpus'],doc_id),[])
            dois = {normalize_doi(m.get('doi',m.get('DOI',''))) for m in metas}; dois.discard('')
            require(len(dois)<=1,'conflicting_legacy_doi_metadata', {'corpus':spec['corpus'],'document_id':doc_id,'dois':sorted(dois)})
            doi=next(iter(dois),'')
            require(isinstance(text,str),'invalid_source_text_type',str(source_id))
            if not text.strip():empty_legacy_texts.append({'corpus':spec['corpus'],'source_id':str(source_id),'document_id':doc_id})
            if not valid_doi(doi) and (spec['corpus'],doc_id) not in {(m['corpus'],m['document_id']) for m in missing_dois}:
                missing_dois.append({'corpus':spec['corpus'],'document_id':doc_id,'invalid_or_missing_doi':doi})
            add_unit(db,corpus=spec['corpus'],split=spec['split'],source_id=source_id,document_id=doc_id,doi=doi if valid_doi(doi) else '',
                     text=text,repr_kind=repr_kind,source_sha=binding['sha256'])
    db.commit(); duplicate_pairs(db); near_pairs(db); db.commit()
    test_clusters=native_test_clusters(db)
    with (destination/'test_source_text_clusters.json.partial').open('x',encoding='utf-8') as output:output.write(json.dumps(test_clusters,ensure_ascii=False,indent=2)+'\n')
    with (destination/'test_source_text_clusters.jsonl.partial').open('x',encoding='utf-8') as output:
        for cluster in test_clusters['clusters']:output.write(json.dumps(cluster,ensure_ascii=False)+'\n')
    def unit(uid):
        r=db.execute('SELECT uid,corpus,split,source_id,document_id,doi,repr_kind,source_sha,byte_sha,projected_sha,norm_sha,word_tokens,gram_count FROM units WHERE uid=?',(uid,)).fetchone()
        return dict(zip(['uid','corpus','split','source_id','document_id','doi','representation','source_sha256','exact_native_bytes_sha256','projected_text_utf8_sha256','normalized_text_sha256','word_tokens','unique_word5grams'],r))
    pair_counts=Counter()
    with (destination/'all_qualifying_pairs.jsonl.partial').open('x',encoding='utf-8') as output:
        for kind,left,right,intersection,union,similarity in db.execute('SELECT * FROM pairs ORDER BY kind,left_uid,right_uid'):
            pair_counts[kind]+=1
            output.write(json.dumps({'kind':kind,'left':unit(left),'right':unit(right),'intersection':intersection,'union':union,'jaccard':similarity,'split_changed':False},ensure_ascii=False)+'\n')
    with (destination/'all_source_units.jsonl.partial').open('x',encoding='utf-8') as output:
        for (uid,) in db.execute('SELECT uid FROM units ORDER BY uid'): output.write(json.dumps(unit(uid),ensure_ascii=False)+'\n')
    short=[unit(uid) for (uid,) in db.execute('SELECT uid FROM units WHERE word_tokens<20 ORDER BY uid')]
    doi_document_groups=[]
    for (doi,) in db.execute("SELECT doi FROM units WHERE doi<>'' GROUP BY doi HAVING COUNT(DISTINCT corpus)>1"):
        rows=list(db.execute('SELECT DISTINCT corpus,split,document_id FROM units WHERE doi=? ORDER BY corpus,split,document_id',(doi,)))
        if any(r[0]=='SC-CoMIcs' for r in rows):
            doi_document_groups.append({'normalized_doi':doi,'source_documents':[dict(zip(['corpus','split','document_id'],r)) for r in rows],'independent_experiments_certified':False})
    counts={}
    for corpus in ['SC-CoMIcs','MuLMS','POLYIE']:
        counts[corpus]={s:dict(zip(['text_units','distinct_source_document_ids'],db.execute('SELECT COUNT(*),COUNT(DISTINCT document_id) FROM units WHERE corpus=? AND split=?',(corpus,s)).fetchone())) for s in NATIVE_SPLITS}
    meta_distribution={}
    for split,ids in NATIVE_SPLITS.items():
        years,types=Counter(),Counter();missing_year,missing_type,missing_meta,conflicting=0,0,0,[]
        for source_id in ids:
            doi=normalize_doi(manifest['normalized_doi_by_id'][str(source_id)]);rows=metadata_by_doi.get(doi,[])
            if not rows:missing_meta+=1
            y={str(r.get('year',r.get('publication_year'))) for r in rows if r.get('year',r.get('publication_year')) not in (None,'')}
            t={str(r.get('article_type',r.get('type'))) for r in rows if r.get('article_type',r.get('type')) not in (None,'')}
            if len(y)>1 or len(t)>1: conflicting.append({'source_id':source_id,'years':sorted(y),'article_types':sorted(t)})
            if len(y)==1:years.update(y)
            else:missing_year+=1
            if len(t)==1:types.update(t)
            else:missing_type+=1
        meta_distribution[split]={'source_ids':len(ids),'missing_official_article_metadata':missing_meta,'missing_or_conflicting_year':missing_year,'missing_or_conflicting_article_type':missing_type,'year_counts':dict(sorted(years.items())),'article_type_counts':dict(sorted(types.items())),'conflicts':conflicting,'temporal_or_article_type_balance_certified':False}
    limitations=[
        'Unique DOI/source IDs do not certify1000 independent original papers, experiments, or pretraining disjointness.',
        'Only already bound official DOI metadata used; absent dates/types/titles are not inferred from IDs/text.',
        'Jaccard measures document-level resemblance of fixed word5gram sets; abstract contained in a long article may have low Jaccard.',
        'MuLMS comparison units are released sentences; POLYIE units are token-list space joins. They are not original article byte reconstructions.',
        'Short text units (<20 word tokens) are not assigned near-duplicate judgments; exact/projected-normalized checks still include them.',
        'No automatic removal, regrouping, split changes, Gold/labels, expert judgment, physical verification, or model evaluation.',
    ]
    result={'status':'synthetic_fixture_audit_completed' if synthetic else 'source_text_provenance_audit_completed_only',
      'synthetic_fixture':synthetic,'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':own_sha,
      'source_only_gate_sha256':gate_sha,'acquisition_manifest_sha256':manifest_sha,'source_stage_protocol_sha256':manifest['protocol_sha256'],
      'native_split_unchanged':manifest['split_ids']==NATIVE_SPLITS,'population_counts':counts,'native_empty_text_ids':empty_texts,
      'legacy_empty_text_units':empty_legacy_texts,
      'unique_normalized_native_doi_identifiers':db.execute("SELECT COUNT(DISTINCT doi) FROM units WHERE corpus='SC-CoMIcs' AND doi<>''").fetchone()[0],
      'declared_unique_native_dois_in_acquisition':manifest.get('unique_normalized_dois'),
      'missing_or_invalid_doi_documents':missing_dois,'short_text_units_near_judgment_unavailable':short,
      'normalized_doi_overlap_source_document_groups':doi_document_groups,
      'legacy_source_files_not_supplied':sorted(set(LEGACY_PATHS)-{str(access.checked(s['path']).relative_to(access.root)) for s in gate.get('legacy_source_inputs',[])}),
      'source_text_scope':{'SC-CoMIcs':'All1000 original abstract.txt records from confirmed native manifest',
        'MuLMS':'Only supplied already exposed parquet sentence/doc_id columns; no omitted article sections reconstructed',
        'POLYIE':'Only supplied already exposed JSON token-list text/id fields, joined with single spaces; no original article layout reconstructed'},
      'pair_counts':dict(pair_counts),'all_qualifying_pairs_retained':True,'fixed_similarity':gate['fixed_similarity'],
      'test_source_text_clusters':test_clusters,
      'official_metadata_distribution_by_native_block':meta_distribution,'input_byte_bindings':bindings,'license_bindings':licenses,
      'declared_native_text_license':gate.get('declared_native_text_license'),'text_archive_binding_from_confirmed_acquisition':{k:v for k,v in manifest['official_archives'][TEXT_ARCHIVE].items() if k!='members'},
      'file_open_log':access.open_log,'annotation_files_opened':False,'annotation_semantics_parsed':False,'network_or_model_api_called':False,
      'model_training_or_inference_performed':False,'checkpoints_or_model_caches_opened':False,'split_changed':False,
      'independent_original_papers_certified':False,'same_experiment_disjointness_certified':False,'pretraining_disjointness_certified':False,
      'physical_truth_or_expert_adjudication_certified':False,'limitations':limitations}
    # Recheck every captured input after all projection/output work, before commit.
    # Gate/manifest rechecks bind the exact JSON bytes actually parsed at startup.
    final_checks=[]
    expected_inputs=bindings+licenses+working_copies+[
        {'path':str(access.checked(source_only_protocol).relative_to(access.root)),'sha256':gate_sha},
        {'path':str(access.checked(acquisition_manifest).relative_to(access.root)),'sha256':manifest_sha}]
    if gate.get('source_stage_protocol_path'):
        expected_inputs.append({'path':gate['source_stage_protocol_path'],'sha256':gate['source_stage_protocol_sha256']})
    for binding in expected_inputs:
        final_checks.append(access.verified(binding['path'],binding['sha256'],binding.get('bytes')))
    require(digest_bytes(Path(__file__).read_bytes())==own_sha,'script_changed_during_audit','Own source bytes changed')
    result['all_captured_inputs_final_rehashed']=True
    result['final_input_byte_checks']=final_checks
    result['source_script_final_sha256']=own_sha
    result['file_open_log']=access.open_log
    # Successful outputs retain metadata/pairs, not raw copies or the working index.
    db.close()
    (destination/'source_text_index.sqlite3').unlink()
    for copied in working_copies:(access.root/copied['path']).unlink()
    if working_copies:(destination/'__source_projections__').rmdir()
    names=['all_qualifying_pairs.jsonl','all_source_units.jsonl','test_source_text_clusters.json','test_source_text_clusters.jsonl']
    output_bindings=[]
    for name in names:
        sha,size=access.hash(destination/(name+'.partial'),'derived_output_byte_hash')
        output_bindings.append({'path':name,'sha256':sha,'bytes':size})
    result['derived_output_byte_bindings']=output_bindings
    with (destination/'audit.json.partial').open('x',encoding='utf-8') as output:output.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    sha,size=access.hash(destination/'audit.json.partial','derived_audit_byte_hash')
    output_bindings=output_bindings+[{'path':'audit.json','sha256':sha,'bytes':size}]
    for name in names+['audit.json']:
        original=destination/(name+'.partial');original.rename(destination/name)
    for binding in output_bindings:
        sha,size=access.hash(destination/binding['path'],'derived_output_post_rename_rehash')
        require(sha==binding['sha256'] and size==binding['bytes'],'output_changed_before_manifest',binding['path'])
    publication={'status':'synthetic_fixture_outputs_complete' if synthetic else 'source_text_only_outputs_complete',
      'synthetic_fixture':synthetic,'input_final_rehash_passed':True,'source_script_sha256':own_sha,
      'gate_sha256':gate_sha,'acquisition_manifest_sha256':manifest_sha,'outputs':output_bindings,
      'annotation_semantics_read':False,'statistical_or_model_analysis_performed':False}
    with (destination/'output_manifest.json').open('x',encoding='utf-8') as output:output.write(json.dumps(publication,indent=2)+'\n')
    return result


def run_audit(root, acquisition_manifest, expected_manifest_sha, source_only_protocol, output_dir, *, synthetic=False):
    attempt={}
    try:
        return _run_audit(root,acquisition_manifest,expected_manifest_sha,source_only_protocol,output_dir,synthetic=synthetic,attempt=attempt)
    except Exception as error:
        if attempt.get('created_output'):
            destination=attempt['created_output']
            # A failure never leaves audit.json/output_manifest as a completion claim.
            for name in ['audit.json','output_manifest.json']:
                path=destination/name
                if path.exists():path.rename(destination/(name+'.failed'))
            failure={'status':'failed_source_text_audit_preserved_no_certificate','exception_type':type(error).__name__,
              'code':getattr(error,'code',None),'detail':getattr(error,'detail',str(error)),
              'completed_certificate':False,'partial_outputs_preserved':True,'synthetic_fixture':synthetic,
              'created_at_utc':datetime.now(timezone.utc).isoformat()}
            with (destination/'failed_attempt.json').open('x',encoding='utf-8') as output:output.write(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--acquisition-manifest',type=Path,required=True)
    parser.add_argument('--acquisition-manifest-sha256',required=True)
    parser.add_argument('--source-only-protocol',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--synthetic-fixture',action='store_true')
    args=parser.parse_args()
    try:
        result=run_audit(args.root,args.acquisition_manifest,args.acquisition_manifest_sha256,args.source_only_protocol,args.output_dir,synthetic=args.synthetic_fixture)
    except AuditError as error:
        print(json.dumps({'status':'blocked_source_only','code':error.code,'detail':error.detail,'actual_audit_completed':False},ensure_ascii=False))
        raise SystemExit(2)
    print(json.dumps({'status':result['status'],'population_counts':result['population_counts'],'pair_counts':result['pair_counts'],'split_changed':False},ensure_ascii=False))


if __name__=='__main__': main()
