# Scientific record recovery

Code and saved-output artifacts for **Diagnosing record recovery bottlenecks in chemical and materials information extraction**, by Dongdong Guo and Jiaxuan Li. This is an unpublished research manuscript prepared for Journal of Cheminformatics; the repository does not imply acceptance.

The evaluation separates local agreement, complete annotated records and the attribution of comparator gains. It preserves adverse outcomes: the SC-CoMIcs aligned head did not establish an adjusted correspondence benefit and was outperformed by the adapted biaffine reference. The original MuLMS baseline recovered zero complete annotated measurement-context objects for all five heads. The current supplements document decoder, calibration, continuation, external-endpoint and confidence analyses, including their adverse results and reused-test limits.

## Current manuscript and supplementary evidence

[Manuscript materials dated 9 October 2026](https://github.com/lijiajia3/scientific-record-recovery/releases/tag/manuscript-20261009) provide the final 30-page main manuscript, 40-page Supplementary Information, official Springer Nature LaTeX sources, six figures in Times New Roman and Additional files 2–6. The five scientific evidence ZIPs retain their original bytes. Each release file has a SHA-256 record; a combined public-materials ZIP is available for one download.

See [the material index and compilation instructions](manuscript/2026-10-09/README.md). The current title, PDFs and availability statements refer to this dated snapshot. The historical v1.0.0 code and weight/cache assets remain available separately. The later diagnostic analyses reuse observed tests; they do not establish a new untouched evaluation or independent training replication. The snapshot release itself runs no new experiments.

## Quick saved-output reproduction

Use Python 3.13 and a new output directory:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-replay.txt
# Download replay-data.zip from the v1.0.0 release, then:
unzip replay-data.zip -d replay-data
python replay.py --data replay-data --out replay_reports
```

No API key, paid inference, GPU or checkpoint loading is needed. This command verifies the public manifest, recomputes all nine POLYIE graphs and fifteen MuLMS relation graphs, replays all 1,500 SC-CoMIcs detector/head graphs using the autonomous native scorer and independent Fraction-based statistics, and verifies all thirteen pilot arm counts. The original fixed random streams provide deterministic arithmetic replay, not new independent random samples.

The compact replay checks saved-output semantics and fixed statistics. It does **not** repeat the historical checkpoint/cache execution barrier, regenerate predictions or establish physical truth. The original complete scientific packet was independently extracted and replayed before this public curation. `verification/` records the narrower public replay and its limits.

## Release assets

[Release v1.0.0](https://github.com/lijiajia3/scientific-record-recovery/releases/tag/v1.0.0)

- `replay-data.zip`: source texts/annotations, predictions, fixed analysis outputs, original configuration and exact scoring implementations.
- `weights-caches-NN.zip.partMMM`: small byte chunks of independent ZIP volumes containing original fitted epoch checkpoints, frozen source caches and the MIT-licensed MatSciBERT weights. Run `python download_binary_assets.py --out binary_assets` to reconstruct and verify the ZIP volumes. Then extract all volumes into one directory to retain the relative historical/expanded paths. These support inspection and new replication work; the quick replay does not use them.
- `SHA256SUMS.txt`: file hashes; `RELEASE_ASSET_MANIFEST.json` describes exact asset sizes and scope.

Each reconstructed weight/cache ZIP is standalone. Its `.partMMM` chunks must first be concatenated in numeric order; the download helper checks every chunk and the reconstructed ZIP. Keep all volumes when reconstructing the full binary record. Fitting and source-generation implementations are in `original_sources/`; these retain their original names, including failed or superseded preparation helpers. The **default reproduction entry point is `replay.py`**, not an archived controller. No automatic cloud provisioning, paid API call or local training is launched by the replay.

## Population and inference

POLYIE: 14 paper units; supplied-entity groups; three fitted seeds. MuLMS: seven paper units; text-only entity detection and directed relations; three fitted seeds. SC-CoMIcs: native fold 1, 800/100/100 train/dev/test abstracts; 100 test source units, pooling the three fits within each unit. Unsupported native targets remain false negatives. The pilot has prior benchmark exposure and is a separate attribution boundary, not an independent replication of the supervised feature mechanism.

Six historical SC relation/event contrasts use 100,000 whole-abstract exchanges and common 10,000 paired-source bootstrap draws, with Holm-six and the original historically informed Holm-ten sensitivity. Fitted seeds are not independent document samples. The historical fitting did not retrain on test labels, select a test seed or construct an ensemble. The current manuscript and dated supplements disclose subsequent development-selected decoder and calibration analyses and all prior test exposure.

## Licenses and access

Original project code is MIT-licensed. Third-party licenses take precedence for their own material:

- POLYIE release: Apache 2.0 notice retained. This does not relicense the corpus under Creative Commons.
- MuLMS corpus: CC BY-SA 4.0; original included author code: AGPL 3.0.
- SC-CoMIcs version 3 texts: CC BY-NC 3.0; annotations: CC BY 4.0.
- MatSciBERT: MIT, original model card retained.
- The pilot includes released SciERC-derived scientific annotations and ChatExtract data with provenance preserved in its original packet. No clinical or patient-derived dataset is used here.

See `LICENSES/` and the original notices inside `replay-data.zip`. Unlicensed SC-CoMIcs repository code and comparison-journal PDFs are excluded. Third-party data rights are not overridden by the top-level code license. Readers can download the public assets without a GitHub account.

## Citation

Guo D, Li J (2026). Diagnosing record recovery bottlenecks in chemical and materials information extraction. Unpublished manuscript. Manuscript materials: manuscript-20261009; historical code release: v1.0.0. https://github.com/lijiajia3/scientific-record-recovery.

Language-model assistance supported code and manuscript preparation; model-assisted critiques are not external human peer review. Human authors remain responsible for verification.
