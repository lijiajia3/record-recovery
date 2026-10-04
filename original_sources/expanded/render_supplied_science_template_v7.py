"""Populate the user's original Science-family template from a shared Word specification.

This program formats supplied prose and saved figures. It cannot select experiments,
score predictions, infer authorship, or certify compilation or final review.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import re
import shutil
import unicodedata
import math
import fitz

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'manuscript/templates/science_template_v1.1'
CITATION = re.compile(r'\[([0-9]+(?:[\s,–-]+[0-9]+)*)\]')
INLINE = re.compile(r'https?://[^\s]+|\*\*[^*]+\*\*|\*[^*]+\*|\[[0-9]+(?:[\s,–-]+[0-9]+)*\]')


def binding(p):
    p = Path(p)
    assert p.is_file() and not p.is_symlink(), p
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1048576):
            h.update(b)
    return {'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}


def numbers(s):
    answer = []
    for piece in s.split(','):
        piece = piece.strip()
        if re.fullmatch(r'\d+', piece):
            answer.append(int(piece))
        elif m := re.fullmatch(r'(\d+)\s*[–-]\s*(\d+)', piece):
            a, b = map(int, m.groups())
            assert a <= b
            answer.extend(range(a, b + 1))
        else:
            raise ValueError(f'Unrecognized citation: {piece!r}')
    return answer


def canonicalize(spec):
    """Preserve historical keys; assign document numbers in first-citation order."""
    result = copy.deepcopy(spec)
    assert result['schema'] == 'science_template_word_companion_v1'
    refs = {}
    for section in result['sections']:
        for b in section['blocks']:
            if b['type'] == 'reference':
                assert b['number'] not in refs
                refs[b['number']] = b['text']
    order = []
    def remember(text):
        for m in CITATION.finditer(text):
            for n in numbers(m[1]):
                assert n in refs, f'Citation {n} has no supplied reference'
                if n not in order:
                    order.append(n)
    for section in result['sections']:
        for b in section['blocks']:
            if b['type'] != 'reference':
                remember(b.get('text', ''))
                for row in b.get('rows', []):
                    for cell in row:
                        remember(cell)
    # Uncited entries are preserved and explicitly reported; they are never hidden.
    uncited = [n for n in refs if n not in order]
    order += uncited
    mapping = {n: i + 1 for i, n in enumerate(order)}
    def rewrite(text):
        return CITATION.sub(lambda m: '[' + ','.join(str(mapping[n]) for n in numbers(m[1])) + ']', text)
    for section in result['sections']:
        for b in section['blocks']:
            if b['type'] != 'reference' and 'text' in b:
                b['text'] = rewrite(b['text'])
            if 'rows' in b:
                b['rows'] = [[rewrite(cell) for cell in row] for row in b['rows']]
        locations = [i for i, b in enumerate(section['blocks']) if b['type'] == 'reference']
        if locations:
            assert not section.get('supplement'), 'Use one reference list before acknowledgments'
            assert locations == list(range(locations[0], locations[-1] + 1))
            section['blocks'][locations[0]:locations[-1] + 1] = [
                {'type': 'reference', 'number': mapping[n], 'text': refs[n]} for n in order
            ]
    return result, mapping, uncited


LATIN_ACCENTS = {'\u0300': '`', '\u0301': "'", '\u0302': '^', '\u0303': '~',
                 '\u0308': '"', '\u0306': 'u', '\u030c': 'v', '\u0327': 'c', '\u0304': '='}
SYMBOLS = {'<': '$<$', '>': '$>$', '–': '--', '—': '---', '−': '$-$', '×': '$\\times$', '±': '$\\pm$',
           '≤': '$\\leq$', '≥': '$\\geq$', '→': '$\\rightarrow$', '∈': '$\\in$',
           'α': '$\\alpha$', 'β': '$\\beta$', 'γ': '$\\gamma$', 'δ': '$\\delta$',
           'Δ': '$\\Delta$', 'θ': '$\\theta$', 'ε': '$\\epsilon$',
           '²': '$^{2}$', '³': '$^{3}$', '¹': '$^{1}$', '’': "'", '‘': "'", '“': '``', '”': "''"}
SPECIAL = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#',
           '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'}


def escape(text):
    assert '\n' not in text
    out = []
    for ch in text:
        if ch in SYMBOLS:
            out.append(SYMBOLS[ch])
        elif ch in SPECIAL:
            out.append(SPECIAL[ch] + ('\\allowbreak{}' if ch == '_' else ''))
        elif ord(ch) < 128:
            out.append(ch + ('\\allowbreak{}' if ch == '/' else ''))
        else:
            decomposed = unicodedata.normalize('NFD', ch)
            if len(decomposed) == 2 and ord(decomposed[0]) < 128 and decomposed[1] in LATIN_ACCENTS:
                out.append('\\' + LATIN_ACCENTS[decomposed[1]] + '{' + decomposed[0] + '}')
            else:
                raise ValueError(f'Explicit TeX conversion required for {ch!r}')
    return re.sub(r'(?<![A-Za-z0-9])([a-f0-9]{32,64})(?![A-Za-z0-9])', lambda m: '\\allowbreak{}'.join(m[1][i:i+8] for i in range(0,len(m[1]),8)), ''.join(out))


def inline(text):
    out, previous = [], 0
    for m in INLINE.finditer(text):
        out.append(escape(text[previous:m.start()]))
        token = m[0]
        if token.startswith('http'):
            assert not any(c in token for c in '{}\\'), 'Unsafe URL syntax'
            out.append('\\url{' + token + '}')
        elif token.startswith('**'):
            out.append('\\textbf{' + escape(token[2:-2]) + '}')
        elif token.startswith('*'):
            out.append('\\textit{' + escape(token[1:-1]) + '}')
        else:
            out.append('\\cite{' + ','.join('ref' + str(n) for n in numbers(token[1:-1])) + '}')
        previous = m.end()
    out.append(escape(text[previous:]))
    return ''.join(out)


def caption(text, expected, supplement):
    m = re.match(r'^\*\*' + expected + r' (S?\d+)\. (.+?)\*\*(.*)$', text)
    assert m, f'Supply a numbered caption with a bold opening sentence: {text}'
    assert m[1].startswith('S') == supplement
    return int(m[1].lstrip('S')), '\\textbf{' + escape(m[2]) + '}' + inline(m[3])


def estimate_lines(text, width, bold=False):
    f = fitz.Font('tibo' if bold else 'tiro')
    plain = re.sub(r'\*\*|\*', '', text)
    width = max(width, 20)
    line, total = 0., 1
    # Estimates are deliberately conservative; actual PDF inspection still required.
    for word in re.findall(r'[^\s_/]+[\s_/]*', plain):
        measured = f.text_length(word, fontsize=12) * 1.12
        if measured > width:
            total += max(1, math.ceil(measured / width) - 1); line = measured % width
        elif line and line + measured > width: total += 1; line = measured
        else: line += measured
    return total


def table_chunks(rows, widths, cap_text, maximum):
    physical = [w / 20 - 12 for w in widths]
    head = max(estimate_lines(c, w, True) for c,w in zip(rows[0], physical))
    cap = estimate_lines(re.sub(r'\*\*', '', cap_text), 468)
    budget = 648 - 85 - 21.7 * (head + cap)
    assert budget > 80
    chunks=[]; current=[]; height=0.
    for row in rows[1:]:
        h=21.7 * (max(estimate_lines(c,w) for c,w in zip(row, physical)) + .20)
        assert h <= budget, 'A single table row needs a separate presentation panel'
        if current and (height+h > budget or len(current)>=maximum): chunks.append(current);current=[];height=0.
        current.append(row);height+=h
    if current: chunks.append(current)
    return chunks


def render(spec_path, destination):
    assert not destination.exists(), 'Use a fresh project; preserve earlier proofs'
    raw = json.loads(spec_path.read_text())
    spec, mapping, uncited = canonicalize(raw)
    assert len(spec['sections']) == 2
    assert not spec['sections'][0].get('supplement') and spec['sections'][1].get('supplement')
    original = (TEMPLATE / 'science_template.tex').read_text()
    preamble = original.split('%%%%%%%%%%%%%%%% TITLE AND AUTHORS %%%%%%%%%%%%%%%%')[0]
    assert len(re.findall(r'\\usepackage(?:\[[^\]]*\])?\{[^}]+\}', preamble)) == 7
    # All original preamble bytes, including production comments and custom command, retained.
    destination.mkdir()
    assets = destination / 'assets'
    assets.mkdir()
    inputs = {str(spec_path.relative_to(ROOT)): binding(spec_path),
              str(Path(__file__).resolve().relative_to(ROOT)): binding(Path(__file__))}
    for name in ('science_template.tex', 'scicite.sty', 'sciencemag.bst', 'readme.txt'):
        inputs['manuscript/templates/science_template_v1.1/' + name] = binding(TEMPLATE / name)
        if name != 'science_template.tex':
            shutil.copyfile(TEMPLATE / name, destination / name)
    authors = next((b['text'] for b in spec['sections'][0]['blocks'] if b['type'] == 'authors'), 'Author details pending')
    title = next(b['text'] for b in spec['sections'][0]['blocks'] if b['type'] == 'title')
    content = [preamble, '\\newcommand{\\scititle}{' + escape(title) + '}\n',
               '\\title{\\bfseries \\boldmath \\scititle}\n', '\\author{' + escape(authors) + '}\n',
               '\\begin{document}\n\\maketitle\n']
    references = []
    for section_index, section in enumerate(spec['sections']):
        supplement = bool(section.get('supplement'))
        if supplement:
            content += ['\\clearpage\n', '\\renewcommand{\\thefigure}{S\\arabic{figure}}\n',
                        '\\renewcommand{\\thetable}{S\\arabic{table}}\n',
                        '\\renewcommand{\\theequation}{S\\arabic{equation}}\n',
                        '\\renewcommand{\\thepage}{S\\arabic{page}}\n',
                        '\\setcounter{figure}{0}\n\\setcounter{table}{0}\n',
                        '\\setcounter{equation}{0}\n\\setcounter{page}{1}\n',
                        '\\begin{center}\n\\section*{Supplementary Materials for\\\\ \\scititle}\n',
                        escape(authors) + '\n\\end{center}\n']
        blocks = section['blocks']
        first_prose, i = True, 0
        while i < len(blocks):
            b = blocks[i]
            kind = b['type']
            if kind in ('title', 'authors'):
                i += 1
                continue
            if kind == 'reference':
                references.append(b)
            elif kind == 'heading' and b['text'] == 'References and Notes':
                content.append('\\clearpage\n\\bibliography{manuscript_references}\n\\bibliographystyle{sciencemag}\n')
            elif kind == 'heading':
                if not supplement and b['text'] not in ('Acknowledgments', 'Supplementary materials'):
                    assert len(b['text'].split()) <= 6, 'Main heading exceeds six words'
                if b.get('pageBreakBefore'):
                    content.append('\\clearpage\n')
                command = 'subsubsection' if b.get('level') == 2 else 'subsection'
                content.append('\\' + command + '*{' + escape(b['text']) + '}\n')
            elif kind in ('paragraph', 'status'):
                if b.get('bold') and first_prose and not supplement:
                    assert not CITATION.search(b['text']), 'No citations in abstract'
                    content.append('\\begin{abstract}\\bfseries \\boldmath\\begingroup\\emergencystretch=24pt\n' + inline(b['text']) + '\n\\par\\endgroup\\end{abstract}\n')
                else:
                    if first_prose:
                        content.append('\\noindent\n')
                    content.append(inline(b['text']) + '\n\n')
                    first_prose = False
            elif kind == 'figure':
                assert i + 1 < len(blocks) and blocks[i + 1]['type'] == 'caption'
                number, text = caption(blocks[i + 1]['text'], 'Figure', supplement)
                source = ROOT / b['path']
                assert source.is_file()
                inputs[b['path']] = binding(source)
                if source.with_suffix('.pdf').exists():
                    source = source.with_suffix('.pdf')
                    inputs[str(source.relative_to(ROOT))] = binding(source)
                asset_name = f'figure_{"S" if supplement else ""}{number}' + source.suffix
                shutil.copyfile(source, assets / asset_name)
                width = b.get('widthFraction', 1)
                assert 0 < width <= 1
                # Preserve the requested physical width; never shrink labeled figures to hide overflow.
                content.append('\\clearpage\n' if b.get('pageBreakBefore') else '')
                content.append('\\setcounter{figure}{' + str(number - 1) + '}\n\\begin{figure}[p]\n\\centering\n')
                content.append('\\includegraphics[width=' + str(width) + '\\textwidth]{assets/' + asset_name + '}\n')
                img = fitz.open(source) if source.suffix == '.pdf' else None
                if img is not None:
                    height_inches = 6.5 * width * img[0].rect.height / img[0].rect.width; img.close()
                else:
                    from PIL import Image
                    with Image.open(source) as im: height_inches = 6.5 * width * im.height / im.width
                # Budget the full legend at the unchanged template line spacing.
                # Long legends follow the figure when the combined estimate would
                # exceed the page; this does not shrink the plotted labels.
                caption_height = 21.7 * estimate_lines(blocks[i + 1]['text'], 468) + 36
                separate_legend = height_inches > 6.6 or height_inches * 72 + caption_height > 625
                if separate_legend:
                    content.append('\\caption{' + escape('Legend on following page.') + '}\n\\label{fig:' + ('S' if supplement else '') + str(number) + '}\n\\end{figure}\n\\clearpage\n')
                    content.append('\\noindent\\textbf{Figure ' + ('S' if supplement else '') + str(number) + '.} ' + text + '\n\n')
                else:
                    content.append('\\caption{' + text + '}\n\\label{fig:' + ('S' if supplement else '') + str(number) + '}\n\\end{figure}\n')
                i += 1
            elif kind == 'caption' and i + 1 < len(blocks) and blocks[i + 1]['type'] == 'table':
                table = blocks[i + 1]
                number, text = caption(b['text'], 'Table', supplement)
                widths = table['widths']
                assert sum(widths) == 9360 and len(widths) >= 2
                rows = table['rows']
                assert all(len(row) == len(widths) for row in rows)
                assert len(rows) >= 2
                chunk_size = table.get('rowsPerPage', 8)
                assert 1 <= chunk_size <= 15
                # Basic tabular with explicit continued pages; no extra packages in user's template.
                physical_widths = [w / 20 - 12 for w in widths]
                assert all(w > 0 for w in physical_widths)
                columns = ''.join('p{' + f'{w:.3f}' + 'pt}' for w in physical_widths)
                for chunk_index, chunk in enumerate(table_chunks(rows, widths, b['text'], chunk_size)):
                    content.append('\\clearpage\n\\setcounter{table}{' + str(number - 1) + '}\n\\begin{table}[p]\n\\centering\n')
                    content.append('\\caption{' + (text if chunk_index == 0 else 'continued.') + '}\n')
                    content.append('\\begin{tabular}{' + columns + '}\n\\hline\n')
                    content.append(' & '.join('{\\raggedright\\textbf{' + inline(cell) + '}\\par}' for cell in rows[0]) + '\\\\\n\\hline\n')
                    for row in chunk:
                        content.append(' & '.join('{\\raggedright ' + inline(cell) + '\\par}' for cell in row) + '\\\\\n')
                    content.append('\\hline\n\\end{tabular}\n\\end{table}\n')
                i += 1
            else:
                raise ValueError(f'Unpaired or unsupported block: {kind}')
            i += 1
    assert len(references) == len(mapping)
    content.append('\\clearpage\n\\end{document}\n')
    tex = ''.join(content)
    assert tex.startswith(preamble)
    assert re.findall(r'\\usepackage(?:\[[^\]]*\])?\{[^}]+\}', tex) == re.findall(r'\\usepackage(?:\[[^\]]*\])?\{[^}]+\}', preamble)
    (destination / 'scientific_extraction_manuscript.tex').write_text(tex)
    # Already verified complete reference strings are retained literally; no author metadata guessed.
    bibliography = ''.join('@misc{ref' + str(b['number']) + ',\n  note = {{' + inline(b['text']) + '}}\n}\n\n' for b in references)
    (destination / 'manuscript_references.bib').write_text(bibliography)
    (destination / 'canonical_word_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
    mapping_record = {'historical_key_to_document_number': mapping, 'uncited_preserved_historical_keys': uncited,
                      'uncited_entries_require_final_author_review': bool(uncited), 'original_input_changed': False}
    (destination / 'reference_number_mapping.json').write_text(json.dumps(mapping_record, indent=2) + '\n')
    outputs = {str(p.relative_to(destination)): binding(p) for p in destination.rglob('*') if p.is_file()}
    receipt = {'scope': 'Official-template formatting source project; compilation and final manuscript review not certified',
               'source_inputs': inputs, 'outputs': outputs, 'preamble_byte_identity': True,
               'no_added_packages': True, 'one_reference_list': True, 'first_citation_numbering': True,
               'historical_key_mapping': mapping, 'uncited_preserved_historical_keys': uncited,
               'experimental_scoring_or_API_invoked': False, 'compiled_PDF_verified': False}
    (destination / 'formatting_source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('spec', type=Path)
    ap.add_argument('destination', type=Path)
    args = ap.parse_args()
    receipt = render(args.spec.resolve(), args.destination.resolve())
    print(json.dumps({'output': str(args.destination), 'reference_count': len(receipt['historical_key_mapping']),
                      'preamble_byte_identity': True, 'compiled_PDF_verified': False}))
