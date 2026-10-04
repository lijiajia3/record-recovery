# Reproduction scope

The public entry point is `replay.py`. It uses the released native annotations and all selected test-source graphs to reproduce the reported saved-output counts and fixed statistics. The default run performs no checkpoint loading, model fitting, network call or paid inference.

The compact component includes all nine POLYIE graphs, all fifteen MuLMS directed-head graphs (and detector graphs), all 1,500 SC-CoMIcs detector/head source graphs, and the thirteen-arm pilot packet. Original scoring source is retained under its frozen names. The release binary volumes contain all 27 POLYIE, 54 MuLMS and 150 SC-CoMIcs epoch checkpoints, plus frozen source caches and the supplied MatSciBERT weights.

The public curation is narrower than the full internal execution-provenance packet. It does not repeat the pre-scoring checkpoint/cache barrier, publish host/cloud-account logs, or include every intermediate development-probability output. The archived fitting/generation implementations and released training/dev/test data support new replication work with an appropriate backend. They are not automatic reproduction launchers, and new hardware can produce different predictions.

The actual public verification report records a fresh extraction of `replay-data.zip` followed by semantic/statistical replay. The binary report separately checks every decoded ZIP member against the frozen original. GitHub-reported SHA256 digests verify the uploaded byte chunks; this is not a claim of independently regenerating neural outputs or of cross-GPU bitwise equality.

The full study concerns linguistic annotation recovery, not independently adjudicated physical or chemical validity. It preserves stronger directed references and unresolved correspondence/feedback contrasts.
