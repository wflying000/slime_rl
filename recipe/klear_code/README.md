# KlearReasoner code reward for slime

Ports the code reward from
`KlearReasoner/verl/utils/reward_score/code_r1_compute.py` (and its
`code_r1_reward/` helpers) into a slime `--custom-rm-path` function.

## What is preserved from klear

* Scoring constants: `correct=1.0`, `incorrect=-1.0`, `format_error=-1.0`,
  `wo_eos_think=-0.5`, `wo_bos_think=-2.0`.
* The long-CoT format gate.
* The klear-specific code extraction rule
  (`</think>.*```python\n(.*?)\n````).
* Sandbox execution via `firejail` with identical command flags
  (`--private --seccomp=socket --profile=pip --rlimit-*`).
* Per-test-case threading up to `MAX_WORKERS=2`, random sampling of up to
  `MAX_TEST_CASES=16` test cases.
* Continuous score: average of per-test `{correct, incorrect}` results.
* Supported test formats: `inputs`/`outputs` (stdin/stdout),
  `functional` (str or list of test strings), `pytest`,
  `inputs_file`/`outputs_file`.

## Changes for slime

* `async def klear_code_reward(args, sample, **kwargs)` wraps the blocking
  scorer with `asyncio.to_thread` and an overall 180s timeout so the
  rollout loop never stalls.
* When `firejail` is not on `$PATH`, `code_r1_reward/firejail_exec.py`
  falls back to `subprocess.run` with `resource.setrlimit`
  (`RLIMIT_AS`/`RLIMIT_CPU`) and `start_new_session=True`. Isolation is
  **weaker** than firejail — intended for dev/test. Production training
  should install firejail.

## Security

This reward executes untrusted model-generated Python. Use firejail (or an
equivalent container sandbox) on any multi-tenant or internet-connected
host. The subprocess fallback is only a soft guard.

## Data

Klear's training jsonl stores test cases under `reward_model.ground_truth`
(a JSON-encoded string). Flatten to slime format:

```bash
python examples/klear_code/prepare_data.py \
    --input  /Users/a58/RL/data/klear_code_100.jsonl \
    --output /tmp/klear_code_slime.jsonl
```

## Train

```bash
bash examples/klear_code/run_klear_code.sh
```

Matches klear's DAPO code hyperparameters: Qwen3.5-9B, GRPO with
`eps-clip=0.2/0.28`, `lr=5e-7`, `n_samples_per_prompt=8`,
`rollout_batch_size=128`, `max_response_len=32768`, `temperature=1.0`, plus
`check_reward_nonzero_std` to reproduce `filter_groups_metric=acc`.

## Logging

Per-sample `(reward, data_source, response, metadata)` records are appended
to `klear_code_reward.CODE_REWARD_LOG_PATH` (default
`/datas/logs/klear_code_reward_logs.jsonl`). Set the constant to `None` to
disable.
