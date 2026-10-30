# Recovery pair experiment: handoff

Read README.md, DATA_CONTRACT.md and VALIDATION.md first. This is a new independent
experiment, not the old full-model SFT trainer. Do not alter existing datasets,
CAP-X code, RLinf code or old training configs to run it.

- User's current experiment uses the EXISTING four segment classes as training
  labels. Do not require another local-action review to use them. Preserve their
  actual WCM/program-completion provenance; do not fabricate verified_local_actions.
- Reuse existing selection.json and segment boundaries to build four libraries FIRST.
  bank_class supplies training_quality. local_quality/trusted_error_mask may supply
  optional finer evidence; unknown does not block existing-class training.
  Do not require an encoder or calibrated matching to create libraries.
- Split whole parents/initial groups before windowing. Handoff state-only records
  belong in S_vla_all, not the four segment libraries. Post-matching eligible_banks
  must remain separate from the original classified banks.
- Portable CPU entry: `python -m recovery_pair build-libraries --sources-config
  configs/data_sources.example.json --data-root /new/server/root --output /new/output`.
  Dataset paths resolve under data-root; algorithm_config resolves beside source
  config. Never treat A800 absolute output indexes as portable training assets.
- Original expert40 is not a VLA rollout. Do not silently add it to S_vla_all.
- Group by initial scene AND full parent episode before matching/calibration.
- FIRST match CAP-success windows against the frozen VLA-POSITIVE bank only.
  Latest preset uses cap_success_admission=segment_suffix: if a successful program
  has any match, admit its continuous suffix from the EARLIEST matched window start
  to that SAME original program segment's end. Preserve every later recorded action;
  never prepend unmatched actions or extend to other/failed programs. Downstream
  windows are admitted by successful-segment provenance, not individually matched.
  Trace suffix anchor, its VLA reference and exact original row range. Generic
  Config defaults to matched_window for compatibility with previous output assets.
  Bad/unknown/handoff VLA states
  remain in the state inventory but cannot admit CAP successes. Never chain via code.
- Current coverage preset: 5cm/30deg/RMS0.50/image0.007 (both RGB views).
  Strict negative pairing stays 1cm/5deg/RMS0.10/image0.002. User accepts visible
  object-layout differences; old qualitative incompatibility reviews are not vetoes.
- Portable admission entry: scripts/run_success_admission.sh or
  python -m recovery_pair.success_admission. Check complete numeric candidate
  feature coverage before matching; missing features must not silently reject a
  candidate. Success-only outputs do not claim full negative pairing is prepared.
- Latest negative_query_policy=vla_failure_only_v1: ONLY VLA-error windows query
  the positive pool for strict pairing. CAP-failure segments keep their original
  labels in source libraries, but are excluded from the eligible negative pool,
  matching and pair training. Positive references still include VLA positives plus
  admitted CAP-success suffixes. Existing VLA-negative windows use valid action
  elements as the error mask unless finer evidence exists. Record this mask basis.
  Rebuild older prepared assets without this policy marker before training.
- Task progress and progress_scorer_id are optional provenance only. Do not gate
  coverage/pairing or rank candidates by them. Legacy progress thresholds are ignored.
- NO privileged inputs for matching/calibration. Whitelist: original public task
  instruction, starting front/wrist RGB and the exact raw/normalized state8 used by
  pi05. Compute EEF distances from that raw state, not separate EEF sidecars.
  Never gate/index/rank by phase, object IDs/poses, held/contact truth, manual verified
  flags, private task_id, progress or future observations. Only RGB front/wrist views
  may enter image features. Labels select pools; episode/initial metadata prevent leaks.
- Prepared matches must declare observation_policy=policy_inputs_only_v1; regenerate
  old privileged prepared indexes without changing source data.
- Never initialize pair sampling with recorded actions, detach a flow step, or
  replace continuous repulsion with a hinge/cutoff. Pull and push share one mask.
- Do not claim real-model/real-data validation from toy CPU tests.
- No paid API, replay, long training, killing processes or upload is implicit in
  inspecting this folder. Check free GPU memory before any GPU allocation.
- Use the pinned RLinf interface; validate actual checkpoint parity before an update.
- New run outputs are immutable; use a new output path or explicit --resume.
- Manual distributed gradient SUM is intentional. Do not add DDP gradient averaging
  on top. The loss is already divided by GLOBAL branch batch sizes.

Tests: `python -m pytest`; local Gloo test needs IPC/localhost permission.
Synthetic smoke: `python -m recovery_pair smoke --output outputs/new_smoke --updates 2`.
