# GraphRestore V7.1 Data and Identity Audit

- protocol: `graphrestore-v7.1-agenticir-locked`
- created_utc: `2026-08-14T21:06:40Z`
- result: **PASS**
- failures: `0`
- warnings: `0`

## Checks

- **PASS** `config.resolved_paths` — /root/autodl-tmp/aaa/graphrestore/configs/resolved_paths.yaml
- **PASS** `repo.agenticir.exists` — repository exists: /root/autodl-tmp/graph/upstream/AgenticIR
- **PASS** `repo.agenticir.commit` — commit=9640a291480dee3ba8f2974125d4ee9e3440f3d6
- **PASS** `repo.agenticir.remote` — remote=https://github.com/Kaiwen-Zhu/AgenticIR.git
- **PASS** `repo.agenticir.tracked_clean` — no tracked worktree changes
- **PASS** `repo.mioir.exists` — repository exists: /root/autodl-tmp/graph/upstream/MiOIR
- **PASS** `repo.mioir.commit` — commit=4d5f6ca0235cf2c307319673242d5722ee35d73f
- **PASS** `repo.mioir.remote` — remote=https://github.com/Xiangtaokong/MiOIR.git
- **PASS** `repo.mioir.tracked_clean` — no tracked worktree changes
- **PASS** `contract.agenticir_commit` — contract commit=9640a291480dee3ba8f2974125d4ee9e3440f3d6
- **PASS** `contract.mioir_commit` — contract commit=4d5f6ca0235cf2c307319673242d5722ee35d73f
- **PASS** `contract.agenticir_remote` — contract remote=https://github.com/Kaiwen-Zhu/AgenticIR.git
- **PASS** `contract.mioir_remote` — contract remote=https://github.com/Xiangtaokong/MiOIR.git
- **PASS** `contract.stage_a_parent_sha256` — contract sha256=66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8
- **PASS** `contract.manifest.clean_train.sha256` — contract sha256=00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257
- **PASS** `contract.manifest.clean_val.sha256` — contract sha256=88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d
- **PASS** `contract.manifest.primary_train.sha256` — contract sha256=83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071
- **PASS** `contract.manifest.primary_val.sha256` — contract sha256=af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b
- **PASS** `contract.manifest.primary_all.sha256` — contract sha256=f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb
- **PASS** `contract.manifest.mio100_test_1440.sha256` — contract sha256=5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950
- **PASS** `contract.agenticir_file.add_single_degradation.sha256` — locked sha256=c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c
- **PASS** `contract.agenticir_file.degradations_txt.sha256` — locked sha256=1a9bae77190579efe9ec17e8f31e09810cb2361b862c33ce4be25a5e3a04d54d
- **PASS** `contract.agenticir_file.scorer.sha256` — locked sha256=b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830
- **PASS** `contract.agenticir_file.compute_scores.sha256` — locked sha256=ce1a35f9f110a67c4581885f631dae6c283e438bcaf2749199fb9d19fa440548
- **PASS** `contract.agenticir_file.compare_methods.sha256` — locked sha256=a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a
- **PASS** `contract.agenticir_file.requirements.sha256` — locked sha256=3e76d9e7c658ce7df907dc39ea7af8aa36aa2d5fcf5bd6ec91d34c109a9b45e2
- **PASS** `agenticir.file.add_single_degradation.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/dataset/add_single_degradation.py
- **PASS** `agenticir.file.add_single_degradation.sha256` — sha256=c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c
- **PASS** `agenticir.file.degradations_txt.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/dataset/degradations.txt
- **PASS** `agenticir.file.degradations_txt.sha256` — sha256=1a9bae77190579efe9ec17e8f31e09810cb2361b862c33ce4be25a5e3a04d54d
- **PASS** `agenticir.file.scorer.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/utils/scorer.py
- **PASS** `agenticir.file.scorer.sha256` — sha256=b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830
- **PASS** `agenticir.file.compute_scores.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/eval/compute_scores.py
- **PASS** `agenticir.file.compute_scores.sha256` — sha256=ce1a35f9f110a67c4581885f631dae6c283e438bcaf2749199fb9d19fa440548
- **PASS** `agenticir.file.compare_methods.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/eval/compare_methods.py
- **PASS** `agenticir.file.compare_methods.sha256` — sha256=a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a
- **PASS** `agenticir.file.requirements.boundary` — inside locked repo: /root/autodl-tmp/graph/upstream/AgenticIR/installation/requirements.txt
- **PASS** `agenticir.file.requirements.sha256` — sha256=3e76d9e7c658ce7df907dc39ea7af8aa36aa2d5fcf5bd6ec91d34c109a9b45e2
- **PASS** `root.training_data` — /root/autodl-tmp/graph/training_data
- **PASS** `root.mio100_data` — /root/autodl-tmp/graph/data/graphrestore
- **PASS** `manifest.clean_train.exists` — /root/autodl-tmp/graph/training_data/manifests/clean_train.jsonl
- **PASS** `manifest.clean_train.sha256` — sha256=00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257
- **PASS** `manifest.clean_train.jsonl` — decoded 3105 strict JSON objects
- **PASS** `manifest.clean_train.row_count` — rows=3105
- **PASS** `manifest.clean_train.schema` — exact schema with 12 fields
- **PASS** `manifest.clean_train.split` — split_counts={'train': 3105}
- **PASS** `manifest.clean_val.exists` — /root/autodl-tmp/graph/training_data/manifests/clean_val.jsonl
- **PASS** `manifest.clean_val.sha256` — sha256=88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d
- **PASS** `manifest.clean_val.jsonl` — decoded 345 strict JSON objects
- **PASS** `manifest.clean_val.row_count` — rows=345
- **PASS** `manifest.clean_val.schema` — exact schema with 12 fields
- **PASS** `manifest.clean_val.split` — split_counts={'val': 345}
- **PASS** `manifest.primary_train.exists` — /root/autodl-tmp/graph/training_data/manifests/primary_train.jsonl
- **PASS** `manifest.primary_train.sha256` — sha256=83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071
- **PASS** `manifest.primary_train.jsonl` — decoded 14400 strict JSON objects
- **PASS** `manifest.primary_train.row_count` — rows=14400
- **PASS** `manifest.primary_train.schema` — exact schema with 16 fields
- **PASS** `manifest.primary_train.split` — split_counts={'train': 14400}
- **PASS** `manifest.primary_val.exists` — /root/autodl-tmp/graph/training_data/manifests/primary_val.jsonl
- **PASS** `manifest.primary_val.sha256` — sha256=af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b
- **PASS** `manifest.primary_val.jsonl` — decoded 1600 strict JSON objects
- **PASS** `manifest.primary_val.row_count` — rows=1600
- **PASS** `manifest.primary_val.schema` — exact schema with 16 fields
- **PASS** `manifest.primary_val.split` — split_counts={'val': 1600}
- **PASS** `manifest.primary_all.exists` — /root/autodl-tmp/graph/training_data/manifests/primary_all.jsonl
- **PASS** `manifest.primary_all.sha256` — sha256=f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb
- **PASS** `manifest.primary_all.jsonl` — decoded 16000 strict JSON objects
- **PASS** `manifest.primary_all.row_count` — rows=16000
- **PASS** `manifest.primary_all.schema` — exact schema with 16 fields
- **PASS** `manifest.primary_all.split` — split_counts={'train': 14400, 'val': 1600}
- **PASS** `manifest.mio100_test_1440.exists` — /root/autodl-tmp/graph/data/graphrestore/manifests/mio100_test_1440.jsonl
- **PASS** `manifest.mio100_test_1440.sha256` — sha256=5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950
- **PASS** `manifest.mio100_test_1440.jsonl` — decoded 1440 strict JSON objects
- **PASS** `manifest.mio100_test_1440.row_count` — rows=1440
- **PASS** `manifest.mio100_test_1440.schema` — exact schema with 11 fields
- **PASS** `manifest.mio100_test_1440.split` — split_counts={'test': 1440}
- **PASS** `clean.train.content` — 3105 rows, 3105 unique IDs, 6210 existing GT/depth paths
- **PASS** `clean.val.content` — 345 rows, 345 unique IDs, 690 existing GT/depth paths
- **PASS** `clean.split_disjoint` — train=3105, val=345, overlap=0
- **PASS** `primary.train.boundary` — 14400 recipes; groups={'single': 7200, 'A': 7200}; only 8 single + 8 ordered Group-A tasks; no forbidden source
- **PASS** `primary.val.boundary` — 1600 recipes; groups={'single': 800, 'A': 800}; only 8 single + 8 ordered Group-A tasks; no forbidden source
- **PASS** `primary.all.boundary` — 16000 recipes; groups={'single': 8000, 'A': 8000}; only 8 single + 8 ordered Group-A tasks; no forbidden source
- **PASS** `primary.all_exact_union` — primary_all is the exact record union of primary_train and primary_val
- **PASS** `mio100.formal_1440.metadata_boundary` — 1440 metadata rows; groups={'A': 640, 'B': 400, 'C': 400}; 2000 referenced paths stat-only; image files opened=0
- **PASS** `mio100.formal_1440.group_counts` — groups={'A': 640, 'B': 400, 'C': 400}
- **PASS** `manifest.mio100_group_a_test.exists` — /root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_a_test_640.jsonl
- **PASS** `manifest.mio100_group_a_test.sha256` — sha256=0516030efe35167abcd94f9bbe124a74fa5a0d00080d748681b1f47f9cdd2ac7
- **PASS** `manifest.mio100_group_a_test.jsonl` — decoded 640 strict JSON objects
- **PASS** `manifest.mio100_group_a_test.row_count` — rows=640
- **PASS** `manifest.mio100_group_a_test.schema` — exact schema with 11 fields
- **PASS** `manifest.mio100_group_a_test.split` — split_counts={'test': 640}
- **PASS** `mio100.mio100_group_a_test.metadata_boundary` — 640 metadata rows; groups={'A': 640}; 880 referenced paths stat-only; image files opened=0
- **PASS** `mio100.mio100_group_a_test.subset_identity` — exact metadata subset of formal-1440 Group A
- **PASS** `manifest.mio100_group_b_test.exists` — /root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_b_test_400.jsonl
- **PASS** `manifest.mio100_group_b_test.sha256` — sha256=6ddbafc620e45e1f7e64d2baeb44d97a51ba236f3d6caeb61cd191806d73491a
- **PASS** `manifest.mio100_group_b_test.jsonl` — decoded 400 strict JSON objects
- **PASS** `manifest.mio100_group_b_test.row_count` — rows=400
- **PASS** `manifest.mio100_group_b_test.schema` — exact schema with 11 fields
- **PASS** `manifest.mio100_group_b_test.split` — split_counts={'test': 400}
- **PASS** `mio100.mio100_group_b_test.metadata_boundary` — 400 metadata rows; groups={'B': 400}; 600 referenced paths stat-only; image files opened=0
- **PASS** `mio100.mio100_group_b_test.subset_identity` — exact metadata subset of formal-1440 Group B
- **PASS** `manifest.mio100_group_c_test.exists` — /root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_c_test_400.jsonl
- **PASS** `manifest.mio100_group_c_test.sha256` — sha256=11748b99ce6d4a894a956bd44eb432d1915f8fe49c47ad3298d251720cc4ecba
- **PASS** `manifest.mio100_group_c_test.jsonl` — decoded 400 strict JSON objects
- **PASS** `manifest.mio100_group_c_test.row_count` — rows=400
- **PASS** `manifest.mio100_group_c_test.schema` — exact schema with 11 fields
- **PASS** `manifest.mio100_group_c_test.split` — split_counts={'test': 400}
- **PASS** `mio100.mio100_group_c_test.metadata_boundary` — 400 metadata rows; groups={'C': 400}; 700 referenced paths stat-only; image files opened=0
- **PASS** `mio100.mio100_group_c_test.subset_identity` — exact metadata subset of formal-1440 Group C
- **PASS** `mio100.exploration.boundary` — manifest path/hash archived only; rows not read; sha256=3552206cfbf6de8b6cea9b68544a8d918ac121a25db55a6cb9c6fd2b0d04a7d1
- **PASS** `parent.manifest.sha256` — sha256=035e18a56282e7f57206730782d81eb2f286504aa5f26755185db7babd547163
- **PASS** `parent.manifest.selected_path` — selected checkpoint=/root/autodl-tmp/aaa/PromptIR_实验归档汇总_20260813/06_ProVIR修理检查继续修理/provir_完整工作区/artifacts/checkpoints/stage_a/final_backbone.ckpt
- **PASS** `parent.manifest.selected_sha` — selected sha256=66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8
- **PASS** `parent.manifest.no_official_test_selection` — official_test_used=false
- **PASS** `parent.checkpoint.sha256` — sha256=66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8

## Machine-readable facts

```json
{
  "agenticir_files": {
    "add_single_degradation": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/dataset/add_single_degradation.py",
      "sha256": "c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c"
    },
    "compare_methods": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compare_methods.py",
      "sha256": "a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a"
    },
    "compute_scores": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compute_scores.py",
      "sha256": "ce1a35f9f110a67c4581885f631dae6c283e438bcaf2749199fb9d19fa440548"
    },
    "degradations_txt": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/dataset/degradations.txt",
      "sha256": "1a9bae77190579efe9ec17e8f31e09810cb2361b862c33ce4be25a5e3a04d54d"
    },
    "requirements": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/installation/requirements.txt",
      "sha256": "3e76d9e7c658ce7df907dc39ea7af8aa36aa2d5fcf5bd6ec91d34c109a9b45e2"
    },
    "scorer": {
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR/utils/scorer.py",
      "sha256": "b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830"
    }
  },
  "data_boundary": {
    "group_b_or_c_training_rows": 0,
    "mio100_allowed_use_at_this_stage": "manifest path/hash boundary audit only",
    "mio100_exploration_rows_read": 0,
    "mio100_formal_image_files_opened": 0,
    "training_groups": [
      "single",
      "A"
    ],
    "training_sources": [
      "MiOIR-Train clean/depth",
      "AgenticIR official operators"
    ]
  },
  "manifests": {
    "clean_train": {
      "path": "/root/autodl-tmp/graph/training_data/manifests/clean_train.jsonl",
      "rows": 3105,
      "schema": [
        "clean_id",
        "clean_path",
        "clean_sha256",
        "depth_dtype",
        "depth_height",
        "depth_path",
        "depth_width",
        "height",
        "source",
        "split",
        "split_seed",
        "width"
      ],
      "sha256": "00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257",
      "splits": {
        "train": 3105
      }
    },
    "clean_val": {
      "path": "/root/autodl-tmp/graph/training_data/manifests/clean_val.jsonl",
      "rows": 345,
      "schema": [
        "clean_id",
        "clean_path",
        "clean_sha256",
        "depth_dtype",
        "depth_height",
        "depth_path",
        "depth_width",
        "height",
        "source",
        "split",
        "split_seed",
        "width"
      ],
      "sha256": "88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d",
      "splits": {
        "val": 345
      }
    },
    "mio100_group_a_test": {
      "path": "/root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_a_test_640.jsonl",
      "rows": 640,
      "schema": [
        "canonical_lq_path",
        "clean_id",
        "degradations",
        "depth_path",
        "group",
        "gt_path",
        "native_lq_path",
        "sample_id",
        "scale_factor",
        "source",
        "split"
      ],
      "sha256": "0516030efe35167abcd94f9bbe124a74fa5a0d00080d748681b1f47f9cdd2ac7",
      "splits": {
        "test": 640
      }
    },
    "mio100_group_b_test": {
      "path": "/root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_b_test_400.jsonl",
      "rows": 400,
      "schema": [
        "canonical_lq_path",
        "clean_id",
        "degradations",
        "depth_path",
        "group",
        "gt_path",
        "native_lq_path",
        "sample_id",
        "scale_factor",
        "source",
        "split"
      ],
      "sha256": "6ddbafc620e45e1f7e64d2baeb44d97a51ba236f3d6caeb61cd191806d73491a",
      "splits": {
        "test": 400
      }
    },
    "mio100_group_c_test": {
      "path": "/root/autodl-tmp/graph/data/graphrestore/manifests/mio100_group_c_test_400.jsonl",
      "rows": 400,
      "schema": [
        "canonical_lq_path",
        "clean_id",
        "degradations",
        "depth_path",
        "group",
        "gt_path",
        "native_lq_path",
        "sample_id",
        "scale_factor",
        "source",
        "split"
      ],
      "sha256": "11748b99ce6d4a894a956bd44eb432d1915f8fe49c47ad3298d251720cc4ecba",
      "splits": {
        "test": 400
      }
    },
    "mio100_test_1440": {
      "path": "/root/autodl-tmp/graph/data/graphrestore/manifests/mio100_test_1440.jsonl",
      "rows": 1440,
      "schema": [
        "canonical_lq_path",
        "clean_id",
        "degradations",
        "depth_path",
        "group",
        "gt_path",
        "native_lq_path",
        "sample_id",
        "scale_factor",
        "source",
        "split"
      ],
      "sha256": "5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950",
      "splits": {
        "test": 1440
      }
    },
    "primary_all": {
      "path": "/root/autodl-tmp/graph/training_data/manifests/primary_all.jsonl",
      "rows": 16000,
      "schema": [
        "canonical_resize",
        "clean_id",
        "clean_path",
        "clean_sha256",
        "degradations",
        "depth_path",
        "group",
        "lq_model_path",
        "lq_native_path",
        "native_scale",
        "operator_order",
        "operator_params",
        "sample_id",
        "seed",
        "source",
        "split"
      ],
      "sha256": "f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb",
      "splits": {
        "train": 14400,
        "val": 1600
      }
    },
    "primary_train": {
      "path": "/root/autodl-tmp/graph/training_data/manifests/primary_train.jsonl",
      "rows": 14400,
      "schema": [
        "canonical_resize",
        "clean_id",
        "clean_path",
        "clean_sha256",
        "degradations",
        "depth_path",
        "group",
        "lq_model_path",
        "lq_native_path",
        "native_scale",
        "operator_order",
        "operator_params",
        "sample_id",
        "seed",
        "source",
        "split"
      ],
      "sha256": "83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071",
      "splits": {
        "train": 14400
      }
    },
    "primary_val": {
      "path": "/root/autodl-tmp/graph/training_data/manifests/primary_val.jsonl",
      "rows": 1600,
      "schema": [
        "canonical_resize",
        "clean_id",
        "clean_path",
        "clean_sha256",
        "degradations",
        "depth_path",
        "group",
        "lq_model_path",
        "lq_native_path",
        "native_scale",
        "operator_order",
        "operator_params",
        "sample_id",
        "seed",
        "source",
        "split"
      ],
      "sha256": "af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b",
      "splits": {
        "val": 1600
      }
    }
  },
  "mio100": {
    "exploration_archive": {
      "allowed_uses": [
        "read_only_protocol_archive"
      ],
      "image_files_opened": 0,
      "path": "/root/autodl-tmp/graph/data/graphrestore/manifests/mio100_exploration_160.jsonl",
      "rows_read": 0,
      "sha256": "3552206cfbf6de8b6cea9b68544a8d918ac121a25db55a6cb9c6fd2b0d04a7d1"
    },
    "formal_1440": {
      "groups": {
        "A": 640,
        "B": 400,
        "C": 400
      },
      "image_files_opened": 0,
      "low_resolution_rows": 460,
      "referenced_paths_stat_only": 2000,
      "rows": 1440,
      "unique_sample_ids": 1440
    },
    "mio100_group_a_test": {
      "groups": {
        "A": 640
      },
      "image_files_opened": 0,
      "low_resolution_rows": 160,
      "referenced_paths_stat_only": 880,
      "rows": 640,
      "unique_sample_ids": 640
    },
    "mio100_group_b_test": {
      "groups": {
        "B": 400
      },
      "image_files_opened": 0,
      "low_resolution_rows": 100,
      "referenced_paths_stat_only": 600,
      "rows": 400,
      "unique_sample_ids": 400
    },
    "mio100_group_c_test": {
      "groups": {
        "C": 400
      },
      "image_files_opened": 0,
      "low_resolution_rows": 200,
      "referenced_paths_stat_only": 700,
      "rows": 400,
      "unique_sample_ids": 400
    }
  },
  "primary": {
    "all": {
      "forbidden_reference_count": 0,
      "groups": {
        "A": 8000,
        "single": 8000
      },
      "rows": 16000,
      "task_counts": {
        "train:dark": 900,
        "train:dark + noise": 900,
        "train:defocus blur": 900,
        "train:defocus blur + haze": 900,
        "train:defocus blur + jpeg compression artifact": 900,
        "train:haze": 900,
        "train:jpeg compression artifact": 900,
        "train:low resolution": 900,
        "train:motion blur": 900,
        "train:motion blur + dark": 900,
        "train:motion blur + low resolution": 900,
        "train:noise": 900,
        "train:noise + jpeg compression artifact": 900,
        "train:rain": 900,
        "train:rain + haze": 900,
        "train:rain + low resolution": 900,
        "val:dark": 100,
        "val:dark + noise": 100,
        "val:defocus blur": 100,
        "val:defocus blur + haze": 100,
        "val:defocus blur + jpeg compression artifact": 100,
        "val:haze": 100,
        "val:jpeg compression artifact": 100,
        "val:low resolution": 100,
        "val:motion blur": 100,
        "val:motion blur + dark": 100,
        "val:motion blur + low resolution": 100,
        "val:noise": 100,
        "val:noise + jpeg compression artifact": 100,
        "val:rain": 100,
        "val:rain + haze": 100,
        "val:rain + low resolution": 100
      },
      "unique_sample_ids": 16000
    },
    "train": {
      "forbidden_reference_count": 0,
      "groups": {
        "A": 7200,
        "single": 7200
      },
      "rows": 14400,
      "task_counts": {
        "train:dark": 900,
        "train:dark + noise": 900,
        "train:defocus blur": 900,
        "train:defocus blur + haze": 900,
        "train:defocus blur + jpeg compression artifact": 900,
        "train:haze": 900,
        "train:jpeg compression artifact": 900,
        "train:low resolution": 900,
        "train:motion blur": 900,
        "train:motion blur + dark": 900,
        "train:motion blur + low resolution": 900,
        "train:noise": 900,
        "train:noise + jpeg compression artifact": 900,
        "train:rain": 900,
        "train:rain + haze": 900,
        "train:rain + low resolution": 900
      },
      "unique_sample_ids": 14400
    },
    "val": {
      "forbidden_reference_count": 0,
      "groups": {
        "A": 800,
        "single": 800
      },
      "rows": 1600,
      "task_counts": {
        "val:dark": 100,
        "val:dark + noise": 100,
        "val:defocus blur": 100,
        "val:defocus blur + haze": 100,
        "val:defocus blur + jpeg compression artifact": 100,
        "val:haze": 100,
        "val:jpeg compression artifact": 100,
        "val:low resolution": 100,
        "val:motion blur": 100,
        "val:motion blur + dark": 100,
        "val:motion blur + low resolution": 100,
        "val:noise": 100,
        "val:noise + jpeg compression artifact": 100,
        "val:rain": 100,
        "val:rain + haze": 100,
        "val:rain + low resolution": 100
      },
      "unique_sample_ids": 1600
    }
  },
  "repositories": {
    "agenticir": {
      "commit": "9640a291480dee3ba8f2974125d4ee9e3440f3d6",
      "path": "/root/autodl-tmp/graph/upstream/AgenticIR",
      "remote": "https://github.com/Kaiwen-Zhu/AgenticIR.git",
      "tracked_changes": []
    },
    "mioir": {
      "commit": "4d5f6ca0235cf2c307319673242d5722ee35d73f",
      "path": "/root/autodl-tmp/graph/upstream/MiOIR",
      "remote": "https://github.com/Xiangtaokong/MiOIR.git",
      "tracked_changes": []
    }
  },
  "stage_a_parent": {
    "checkpoint_path": "/root/autodl-tmp/aaa/PromptIR_实验归档汇总_20260813/06_ProVIR修理检查继续修理/provir_完整工作区/artifacts/checkpoints/stage_a/final_backbone.ckpt",
    "checkpoint_sha256": "66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8",
    "manifest_path": "/root/autodl-tmp/aaa/PromptIR_实验归档汇总_20260813/06_ProVIR修理检查继续修理/provir_完整工作区/artifacts/manifests/stage_a_final_selection.json",
    "manifest_sha256": "035e18a56282e7f57206730782d81eb2f286504aa5f26755185db7babd547163",
    "official_test_used": false,
    "selected": "post_finetune_endpoint"
  }
}
```
