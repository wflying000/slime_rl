# slime RLVR Recipes

[English](README.md) | [简体中文](README_zh.md)

This project is based on [THUDM/slime](https://github.com/THUDM/slime). It provides reward computation and RLVR (Reinforcement Learning with Verifiable Rewards) training scripts for **instruction following, math, coding, and logical reasoning**, and implements **NGRPO** and **SAPO** in the slime training pipeline.

Task implementations live primarily in [`recipe/`](recipe/), while algorithm implementations and arguments live in [`slime/utils/`](slime/utils/). Training uses slime's Megatron-LM, SGLang, and Ray architecture for both single-task and mixed-task training. For the full framework capabilities, environment requirements, and general arguments, see the [slime documentation](https://thudm.github.io/slime/).

## Features

- **Rewards for multiple tasks**: instruction constraint checks, math answer verification, code test execution, and scoring for several logical reasoning tasks.
- **Unified reward entry point**: dispatches scorers using `metadata.data_type`, allowing different tasks in the same training batch.
- **Training recipes**: single-node H800 colocate configurations for Qwen3.5-9B and Qwen3.5-35B-A3B, covering instruction following, math, coding, and mixed tasks.
- **NGRPO**: calibrates group-relative advantages with virtual maximum rewards, preserving negative training signals for all-wrong groups.
- **SAPO**: replaces PPO hard clipping with a token-level soft-gated policy objective, with separate temperatures for positive and nonpositive advantages.
- **Reward timeout and error handling**: runs scorers in terminable CPU workers and excludes invalid samples from group statistics and training loss through reward post-processing.

## Tasks and Rewards

The table lists the main task routes whose implementations are included in this repository. Set `metadata.data_type` to the exact route value shown below; `math`, `coding`, and `logic` are task categories, not aliases accepted by the unified entry point.

| Task | `metadata.data_type` | Scoring | Implementation |
| --- | --- | --- | --- |
| Instruction Following | `if` | Checks IFBench / IFEval-style instruction constraints; returns 1 if all constraints pass, otherwise 0 | [if_reward.py](recipe/instruction_following/if_reward.py) |
| Content-Constrained Instruction Following | `if_cc` | Checks hard constraints, then combines content rules with a semantic judge | [reward.py](recipe/instruction_following_content_constrained/reward.py) |
| Math | `klear_math` | Extracts and verifies the answer; returns 1 for correctness, otherwise 0; prefers `math-verify`, without extra penalties for thinking tags or formatting | [klear_math_reward.py](recipe/klear_math/klear_math_reward.py) |
| Coding | `klear_code` | Extracts Python code and runs test cases; returns 1 only if all tests pass; supports assertions, pytest, and stdin/stdout tests | [klear_code_reward.py](recipe/klear_code/klear_code_reward.py) |
| Guru-RL Logical Reasoning | `guru_rl` | Selects scoring for Zebra grids, graph logic, ordering, and other tasks using `metadata.data_source`; some tasks award partial credit | [guru_rl_reward.py](recipe/guru_rl/guru_rl_reward.py) |
| Reasoning Gym | `nemotron_rl_reasoning_gym` | Selects task verifiers using `metadata.source_dataset`, with string, numeric, or math matching as fallbacks | [reasoning_gym_reward.py](recipe/nemotron_rl_reasoning_gym/reasoning_gym_reward.py) |
| ZebraLogic Multiple Choice | `math_logic_puzzle` | Extracts the choice index from the last complete `\boxed{...}`; returns 1 for correctness, otherwise 0 | [math_logic_puzzle_zebralogic_reward.py](recipe/math_logic_puzzle_zebralogic/math_logic_puzzle_zebralogic_reward.py) |
| ZebraLogicBench | `zebralogicbench` | Uses cell-level accuracy in grid mode and answer matching in multiple-choice mode | [zebralogicbench_reward.py](recipe/zebralogicbench/zebralogicbench_reward.py) |
| Structured Outputs | `nemotron_structured_outputs` | Parses formats including JSON, YAML, XML, TOML, and CSV and validates structural constraints | [structured_outputs_reward.py](recipe/nemotron_structured_outputs/structured_outputs_reward.py) |

The unified entry point is [`recipe.rl_reward.joint_reward.joint_reward`](recipe/rl_reward/joint_reward.py). The dispatcher also registers other task names, but some corresponding modules are not included in this repository. When adding training data, ensure that the selected route's implementation and dependencies are available.

## Repository Structure

```text
recipe/
├── instruction_following/                 # Instruction rewards and constraint checkers
├── instruction_following_content_constrained/ # Content constraints and semantic judge
├── klear_math/                            # Math answer verification
├── klear_code/                            # Code extraction, tests, and sandbox execution
├── guru_rl/                               # Guru-RL task scoring
├── nemotron_rl_reasoning_gym/              # Reasoning Gym task scoring
├── math_logic_puzzle_zebralogic/           # ZebraLogic multiple-choice scoring
├── zebralogicbench/                       # ZebraLogicBench grid and multiple-choice scoring
├── nemotron_structured_outputs/           # Structured-output validation
├── rl_reward/                            # Dispatch, workers, timeouts, and post-processing
├── scripts/                              # Single-task and mixed-task training recipes
└── data/nltk_data/                        # NLTK resources for instruction checks
slime/utils/
├── ngrpo.py                              # NGRPO advantages and reward post-processing
├── sapo.py                               # SAPO policy loss and argument validation
└── arguments.py                          # Training and algorithm arguments
slime/backends/megatron_utils/loss.py      # Algorithm integration into Megatron training
misc/setup_slime.sh                       # Project and reward dependency installation
train.py                                 # Synchronous training entry point
train_async.py                           # slime asynchronous training entry point
```

## Environment Setup

Use the upstream slime Linux / NVIDIA GPU container with compatible Megatron-LM and SGLang environments. See the [upstream quick start](docs/en/get_started/quick_start.md) for environment setup, model weight conversion, and basic training. The recipes use single-node H800 configurations as a starting point; adjust parallelism, memory settings, and batch sizes for your hardware.

From the repository root, inside a container with slime's base dependencies already installed, run:

```bash
bash misc/setup_slime.sh
```

The script installs the project in editable mode (`pip install -e . --no-deps`), installs `math-verify`, `nltk`, `emoji`, `syllapy`, `langdetect`, and `wandb==0.27.2`, and installs `firejail` through `apt-get`, which requires root privileges. It does not install the complete GPU training environment. NLTK resources for instruction following are included in `recipe/data/nltk_data/`.

Add dependencies and configuration for the tasks you use: code rewards use `firejail` to execute generated Python, and pytest test cases require `pytest`. Structured outputs require `jsonschema` and `PyYAML`, plus `tomli` on Python 3.10. For the content-constrained semantic judge, see [llm_content_reward.py](recipe/instruction_following_content_constrained/llm_content_reward.py). The Guru-RL `stem_web` branch requires `STEM_LLM_JUDGE_URL`.

## Data and Reward Integration

Training recipes read JSONL samples using these fields:

| Field | Purpose |
| --- | --- |
| `prompt` | Model input; when applying a chat template, this can be a message list with `role` and `content` |
| `label` | Math answer, code test cases, or a logical reasoning reference answer, in the format required by the scorer |
| `metadata` | Includes at least `data_type`, plus task-specific constraints or source information |

For example, a math sample can be written as:

```json
{"prompt":[{"role":"user","content":"What is 1 + 1?"}],"label":"2","metadata":{"data_type":"klear_math"}}
```

Instruction-following samples also need `metadata.instruction_id_list` and a corresponding `metadata.instruction_kwargs` entry for each instruction. Code samples can store a test-case dictionary as a JSON-encoded string in `label`, such as `{"functional":"assert add(1, 2) == 3"}`. Prepare reference answers and metadata for logical reasoning tasks according to their reward modules.

All tasks can use the following arguments, which are already included in the training recipes:

```bash
--input-key prompt \
--label-key label \
--metadata-key metadata \
--custom-rm-path recipe.rl_reward.joint_reward.joint_reward \
--custom-reward-post-process-path recipe.rl_reward.reward_post_process.reward_post_process
```

`JOINT_REWARD_SAMPLE_TIMEOUT_S` controls the per-sample reward timeout: the worker default is 240 seconds, and mixed-task recipes default to 600 seconds. `JOINT_REWARD_WORKERS` controls concurrency in each reward worker pool and defaults to 8. Invalid routes, scoring errors, and timeouts flag samples for exclusion; the post-processing function masks their losses and excludes them from group statistics. When launching through Ray, pass custom environment variables through the script's `RUNTIME_ENV_JSON.env_vars`.

## Training Recipes

[`recipe/scripts/`](recipe/scripts/) provides single-node colocate configurations, where training and rollout share GPUs. The scripts default to 8 GPUs.

| Task / Algorithm | Qwen3.5-9B | Qwen3.5-35B-A3B |
| --- | --- | --- |
| Instruction Following / GRPO | [9B recipe](recipe/scripts/train_rl_if_1node_colocate_9B_grpo_h800.sh) | [35B-A3B recipe](recipe/scripts/train_rl_if_1node_colocate_35BA3B_grpo_h800.sh) |
| Math / GRPO | [9B recipe](recipe/scripts/train_rl_math_1node_colocate_9B_grpo_h800.sh) | [35B-A3B recipe](recipe/scripts/train_rl_math_1node_colocate_35BA3B_grpo_h800.sh) |
| Coding / GRPO | [9B recipe](recipe/scripts/train_rl_code_1node_colocate_9B_grpo_h800.sh) | [35B-A3B recipe](recipe/scripts/train_rl_code_1node_colocate_35BA3B_grpo_h800.sh) |
| IF + Math + Coding + Logic / GRPO | [9B recipe](recipe/scripts/train_rl_mix_1node_colocate_9B_grpo_h800.sh) | [35B-A3B recipe](recipe/scripts/train_rl_mix_1node_colocate_35BA3B_grpo_h800.sh) |
| Mixed Tasks / NGRPO | [9B recipe](recipe/scripts/train_rl_mix_1node_colocate_9B_ngrpo_h800.sh) | — |
| Mixed Tasks / SAPO | [9B recipe](recipe/scripts/train_rl_mix_1node_colocate_9B_sapo_h800.sh) | — |

Logic tasks use the mixed-task recipes; there is currently no standalone `train_rl_logic_*` script. Structured outputs have a separate [single-node training example](recipe/nemotron_structured_outputs/train_nemotron_structured_outputs_1node_colocate.sh).

Before launching, prepare Hugging Face weights and converted Megatron checkpoints following the quick start. Update `MODEL_DIR`, `DATA_DIR`, `HF_CHECKPOINT`, `REF_CHECKPOINT`, `TRAIN_CHECKPOINT`, `TRAIN_DATAPATH`, and evaluation data paths in the selected script, and check the Megatron-LM path, log directory, and W&B configuration. `/model/dir` and `/data/dir` are placeholders; datasets and model weights must be prepared separately. Model and data paths in the 9B mixed-task scripts are assigned directly, so edit those scripts to change them.

After configuration, choose a script to run from the repository root, for example:

```bash
# Instruction-following GRPO
bash recipe/scripts/train_rl_if_1node_colocate_9B_grpo_h800.sh

# NGRPO with all four task categories
bash recipe/scripts/train_rl_mix_1node_colocate_9B_ngrpo_h800.sh

# SAPO with all four task categories
bash recipe/scripts/train_rl_mix_1node_colocate_9B_sapo_h800.sh
```

## NGRPO and SAPO

### NGRPO

[`slime/utils/ngrpo.py`](slime/utils/ngrpo.py) implements Negative-enhanced GRPO. On the rollout side, before data-parallel redistribution, it augments each prompt group's reward statistics with virtual maximum rewards, then calculates advantages for real samples. Virtual samples contribute only to the mean and standard deviation, never to the training batch. Groups whose real answers are all wrong still receive negative advantages when their rewards are below the maximum.

The example recipe uses:

```bash
--advantage-estimator ngrpo \
--ngrpo-max-reward 1.0 \
--ngrpo-num-virtual-samples 1 \
--ngrpo-std-correction 1
```

Set `--ngrpo-max-reward` to the task's actual maximum reward. `--ngrpo-std-correction 1` uses the sample standard deviation; `0` uses the population standard deviation. NGRPO requires reward and group standard-deviation normalization, with `--normalize-advantages` disabled and `--kl-coef 0`. For optional KL regularization, use `--use-kl-loss` with `--kl-loss-coef`. Keep all-wrong groups in the training data so their training signal is preserved.

### SAPO

[`slime/utils/sapo.py`](slime/utils/sapo.py) implements token-level Soft Adaptive Policy Optimization and integrates it into the [Megatron policy loss](slime/backends/megatron_utils/loss.py). The SAPO recipe uses GRPO group-relative advantages and sigmoid soft gating to modulate policy updates.

```bash
--advantage-estimator grpo \
--policy-loss-type sapo \
--sapo-tau-pos 1.0 \
--sapo-tau-neg 1.05
```

SAPO requires the Megatron training backend, `policy_loss`, at least 2 samples per prompt, and reward and group standard-deviation normalization, with `--normalize-advantages` disabled. Positive and nonpositive advantages use `tau_pos` and `tau_neg`, respectively. PPO clipping thresholds only affect the `sapo_would_clipfrac` diagnostic; they do not hard-clip the SAPO objective.

Algorithm tests are in [test_ngrpo.py](tests/test_ngrpo.py) and [test_sapo_loss.py](tests/test_sapo_loss.py).

## Upstream Resources and License

- [slime GitHub](https://github.com/THUDM/slime)
- [slime documentation](https://thudm.github.io/slime/)
- [Custom reward guide](docs/en/get_started/customization.md#--custom-rm-path)
- [Preserved upstream English README](README_origin.md)
- [Preserved upstream Chinese README](README_zh_origin.md)

This project follows slime's [Apache License 2.0](LICENSE).
