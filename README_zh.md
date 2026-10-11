# slime RLVR Recipes

[English](README.md) | [简体中文](README_zh.md)

本项目基于 [THUDM/slime](https://github.com/THUDM/slime)，面向 **Instruction Following、Math、Coding 和 Logic 推理**提供 reward 计算与 RLVR（Reinforcement Learning with Verifiable Rewards）训练脚本，并在 slime 训练流程中实现了 **NGRPO** 和 **SAPO** 算法。

任务相关实现主要位于 [`recipe/`](recipe/)，算法与参数实现主要位于 [`slime/utils/`](slime/utils/)。训练沿用 slime 的 Megatron-LM、SGLang 和 Ray 架构，支持单任务训练及多任务混合训练。完整的框架能力、环境要求和通用参数请参阅 [slime 官方文档](https://thudm.github.io/slime/)。

## 主要功能

- **多任务 reward**：指令约束检查、数学答案验证、代码测试执行，以及多种逻辑推理任务评分。
- **统一 reward 入口**：根据 `metadata.data_type` 分发评分函数，可在同一批训练数据中混合不同任务。
- **训练脚本**：提供 Qwen3.5-9B 和 Qwen3.5-35B-A3B 的单节点 H800 colocate 配置，涵盖指令遵循、数学、代码和混合任务。
- **NGRPO**：通过虚拟最高奖励样本校准组内优势，使全错组仍保留负向训练信号。
- **SAPO**：使用 token 级 soft gating 策略目标替代 PPO 硬裁剪，提供正、负优势各自的温度参数。
- **reward 超时与异常处理**：在可终止的 CPU worker 中评分，并通过 reward 后处理将无效样本排除出组内统计和训练损失。

## 任务与 Reward

以下为当前仓库中包含实现的主要任务路由。`metadata.data_type` 必须使用表中的具体值；`math`、`coding`、`logic` 只是任务类别名称，不是统一入口支持的路由别名。

| 任务 | `metadata.data_type` | 评分方式 | 实现 |
| --- | --- | --- | --- |
| Instruction Following | `if` | 检查 IFBench / IFEval 风格的指令约束，全部满足得 1，否则得 0 | [if_reward.py](recipe/instruction_following/if_reward.py) |
| 内容约束指令遵循 | `if_cc` | 先检查硬约束，再结合内容规则和语义 judge 评分 | [reward.py](recipe/instruction_following_content_constrained/reward.py) |
| Math | `klear_math` | 提取答案并验证正确性，正确得 1，否则得 0；优先使用 `math-verify`，不额外施加思考标签或格式惩罚 | [klear_math_reward.py](recipe/klear_math/klear_math_reward.py) |
| Coding | `klear_code` | 提取 Python 代码并执行测试用例，全部通过得 1，否则得 0；支持断言、pytest 和 stdin/stdout 测试 | [klear_code_reward.py](recipe/klear_code/klear_code_reward.py) |
| Guru-RL 逻辑推理 | `guru_rl` | 按 `metadata.data_source` 选择 Zebra 网格、图逻辑、排序等任务的评分方法，部分任务支持部分得分 | [guru_rl_reward.py](recipe/guru_rl/guru_rl_reward.py) |
| Reasoning Gym | `nemotron_rl_reasoning_gym` | 按 `metadata.source_dataset` 选择任务验证器，其他任务回退到字符串、数值或数学匹配 | [reasoning_gym_reward.py](recipe/nemotron_rl_reasoning_gym/reasoning_gym_reward.py) |
| ZebraLogic 选择题 | `math_logic_puzzle` | 从最后一个完整的 `\boxed{...}` 中提取选项编号，正确得 1，否则得 0 | [math_logic_puzzle_zebralogic_reward.py](recipe/math_logic_puzzle_zebralogic/math_logic_puzzle_zebralogic_reward.py) |
| ZebraLogicBench | `zebralogicbench` | 网格模式按单元格准确率评分，选择题模式按答案匹配评分 | [zebralogicbench_reward.py](recipe/zebralogicbench/zebralogicbench_reward.py) |
| 结构化输出 | `nemotron_structured_outputs` | 解析 JSON、YAML、XML、TOML、CSV 等输出，并验证结构约束 | [structured_outputs_reward.py](recipe/nemotron_structured_outputs/structured_outputs_reward.py) |

统一入口是 [`recipe.rl_reward.joint_reward.joint_reward`](recipe/rl_reward/joint_reward.py)。分发器还注册了其他任务名称，但部分对应模块未包含在当前仓库中；新增训练数据时，应确认所选路由的实现和依赖均已提供。

## 目录结构

```text
recipe/
├── instruction_following/                 # 指令遵循 reward 与约束检查器
├── instruction_following_content_constrained/ # 内容约束与语义 judge
├── klear_math/                            # 数学答案验证
├── klear_code/                            # 代码提取、测试执行与沙箱
├── guru_rl/                               # Guru-RL 任务评分
├── nemotron_rl_reasoning_gym/              # Reasoning Gym 任务评分
├── math_logic_puzzle_zebralogic/           # ZebraLogic 选择题评分
├── zebralogicbench/                       # ZebraLogicBench 网格与选择题评分
├── nemotron_structured_outputs/           # 结构化输出验证
├── rl_reward/                            # 统一分发、worker、超时与后处理
├── scripts/                              # 单任务与混合任务训练脚本
└── data/nltk_data/                        # 指令检查使用的 NLTK 数据
slime/utils/
├── ngrpo.py                              # NGRPO 优势计算与 reward 后处理
├── sapo.py                               # SAPO 策略损失与参数校验
└── arguments.py                          # 训练及算法参数
slime/backends/megatron_utils/loss.py      # 算法在 Megatron 训练端的接入
misc/setup_slime.sh                       # 项目与 reward 依赖安装
train.py                                 # 同步训练入口
train_async.py                           # slime 异步训练入口
```

## 环境准备

建议使用上游 slime 提供的 Linux / NVIDIA GPU 容器及兼容的 Megatron-LM、SGLang 环境。环境配置、模型权重转换和基础训练流程见 [上游快速开始](docs/zh/get_started/quick_start.md)。本仓库的训练脚本以 H800 单节点配置为起点，实际并行度、显存参数和 batch size 需按硬件调整。

在已经准备好 slime 基础依赖的容器中，从仓库根目录执行：

```bash
bash misc/setup_slime.sh
```

该脚本以 editable 模式安装项目（`pip install -e . --no-deps`），安装 `math-verify`、`nltk`、`emoji`、`syllapy`、`langdetect` 和 `wandb==0.27.2`，并通过 `apt-get` 安装 `firejail`，因此需要 root 权限。它不会自动安装完整的 GPU 训练环境。指令遵循所需的 NLTK 数据已放在 `recipe/data/nltk_data/`。

按所用任务补充依赖与配置：代码 reward 使用 `firejail` 执行生成的 Python；pytest 格式的用例需要 `pytest`。结构化输出需要 `jsonschema`、`PyYAML`，Python 3.10 还需要 `tomli`。内容约束任务的语义 judge 配置见 [llm_content_reward.py](recipe/instruction_following_content_constrained/llm_content_reward.py)；Guru-RL 的 `stem_web` 分支需要配置 `STEM_LLM_JUDGE_URL`。

## 数据与 Reward 接入

训练脚本使用 JSONL 数据，并通过以下字段读取样本：

| 字段 | 用途 |
| --- | --- |
| `prompt` | 模型输入；启用 chat template 时可使用包含 `role`、`content` 的消息列表 |
| `label` | 数学答案、代码测试用例或逻辑任务的参考答案，格式由具体 reward 决定 |
| `metadata` | 至少包含 `data_type`，并提供该任务需要的约束或数据来源信息 |

例如，一条数学样本可以写成：

```json
{"prompt":[{"role":"user","content":"What is 1 + 1?"}],"label":"2","metadata":{"data_type":"klear_math"}}
```

指令遵循样本还需提供 `metadata.instruction_id_list` 和一一对应的 `metadata.instruction_kwargs`；代码样本可将测试用例字典编码为 JSON 字符串放入 `label`，例如 `{"functional":"assert add(1, 2) == 3"}`。逻辑任务应按对应 reward 模块准备参考答案与任务元数据。

所有任务可共用以下参数；训练脚本已包含这些配置：

```bash
--input-key prompt \
--label-key label \
--metadata-key metadata \
--custom-rm-path recipe.rl_reward.joint_reward.joint_reward \
--custom-reward-post-process-path recipe.rl_reward.reward_post_process.reward_post_process
```

`JOINT_REWARD_SAMPLE_TIMEOUT_S` 控制单样本 reward 超时，worker 默认值为 240 秒，混合训练脚本默认设为 600 秒；`JOINT_REWARD_WORKERS` 控制每个 reward worker 池的并发数，默认 8。无效路由、评分异常或超时样本会被标记，由上述后处理函数屏蔽其损失并排除出组内统计。通过 Ray 启动时，自定义环境变量需要传入脚本的 `RUNTIME_ENV_JSON.env_vars`。

## 训练脚本

[`recipe/scripts/`](recipe/scripts/) 提供单节点 colocate 训练配置，训练与 rollout 共用 GPU。脚本默认使用 8 张 GPU。

| 任务 / 算法 | Qwen3.5-9B | Qwen3.5-35B-A3B |
| --- | --- | --- |
| Instruction Following / GRPO | [9B 脚本](recipe/scripts/train_rl_if_1node_colocate_9B_grpo_h800.sh) | [35B-A3B 脚本](recipe/scripts/train_rl_if_1node_colocate_35BA3B_grpo_h800.sh) |
| Math / GRPO | [9B 脚本](recipe/scripts/train_rl_math_1node_colocate_9B_grpo_h800.sh) | [35B-A3B 脚本](recipe/scripts/train_rl_math_1node_colocate_35BA3B_grpo_h800.sh) |
| Coding / GRPO | [9B 脚本](recipe/scripts/train_rl_code_1node_colocate_9B_grpo_h800.sh) | [35B-A3B 脚本](recipe/scripts/train_rl_code_1node_colocate_35BA3B_grpo_h800.sh) |
| IF + Math + Coding + Logic / GRPO | [9B 脚本](recipe/scripts/train_rl_mix_1node_colocate_9B_grpo_h800.sh) | [35B-A3B 脚本](recipe/scripts/train_rl_mix_1node_colocate_35BA3B_grpo_h800.sh) |
| 混合任务 / NGRPO | [9B 脚本](recipe/scripts/train_rl_mix_1node_colocate_9B_ngrpo_h800.sh) | — |
| 混合任务 / SAPO | [9B 脚本](recipe/scripts/train_rl_mix_1node_colocate_9B_sapo_h800.sh) | — |

Logic 任务通过混合训练脚本接入，目前没有独立的 `train_rl_logic_*` 脚本。结构化输出另有 [单节点训练示例](recipe/nemotron_structured_outputs/train_nemotron_structured_outputs_1node_colocate.sh)。

启动前，先按快速开始文档准备 Hugging Face 权重及转换后的 Megatron checkpoint，再修改所选脚本中的 `MODEL_DIR`、`DATA_DIR`、`HF_CHECKPOINT`、`REF_CHECKPOINT`、`TRAIN_CHECKPOINT`、`TRAIN_DATAPATH` 和评测数据路径，同时确认 Megatron-LM 路径、日志目录与 W&B 配置。脚本中的 `/model/dir`、`/data/dir` 是占位路径；数据集和模型权重需另行准备。9B 混合脚本中的模型和数据路径为直接赋值，需要编辑脚本。

配置完成后，从仓库根目录选择一个脚本执行，例如：

```bash
# 指令遵循 GRPO
bash recipe/scripts/train_rl_if_1node_colocate_9B_grpo_h800.sh

# 四类任务混合 NGRPO
bash recipe/scripts/train_rl_mix_1node_colocate_9B_ngrpo_h800.sh

# 四类任务混合 SAPO
bash recipe/scripts/train_rl_mix_1node_colocate_9B_sapo_h800.sh
```

## NGRPO 与 SAPO

### NGRPO

[`slime/utils/ngrpo.py`](slime/utils/ngrpo.py) 实现 Negative-enhanced GRPO。它在 rollout 侧、数据并行分组前，为每个 prompt 组添加虚拟最高奖励用于计算均值与标准差，再计算真实样本的优势。虚拟样本只参与统计，不进入训练 batch；当组内真实答案全部错误且奖励低于最高奖励时，仍能得到负优势。

示例脚本使用以下参数：

```bash
--advantage-estimator ngrpo \
--ngrpo-max-reward 1.0 \
--ngrpo-num-virtual-samples 1 \
--ngrpo-std-correction 1
```

`--ngrpo-max-reward` 应设为任务实际最高奖励；`--ngrpo-std-correction 1` 使用样本标准差，设为 `0` 则使用总体标准差。NGRPO 要求保留 reward 和组内标准差归一化，不启用 `--normalize-advantages`，并保持 `--kl-coef 0`；如需 KL 正则，可使用 `--use-kl-loss` 与 `--kl-loss-coef`。不要过滤全错组，否则会丢失这一算法所保留的训练信号。

### SAPO

[`slime/utils/sapo.py`](slime/utils/sapo.py) 实现 token 级 Soft Adaptive Policy Optimization，并接入 [Megatron 策略损失](slime/backends/megatron_utils/loss.py)。当前 SAPO 示例使用 GRPO 组内优势，通过 sigmoid soft gating 调节策略更新。

```bash
--advantage-estimator grpo \
--policy-loss-type sapo \
--sapo-tau-pos 1.0 \
--sapo-tau-neg 1.05
```

SAPO 要求 Megatron 训练后端、`policy_loss`、每个 prompt 至少 2 个样本，并保留 reward 和组内标准差归一化，不启用 `--normalize-advantages`。正、非正优势分别使用 `tau_pos`、`tau_neg`；PPO 裁剪阈值仅用于 `sapo_would_clipfrac` 诊断，不对 SAPO 目标进行硬裁剪。

算法测试分别位于 [test_ngrpo.py](tests/test_ngrpo.py) 和 [test_sapo_loss.py](tests/test_sapo_loss.py)。

## 上游资源与许可证

- [slime GitHub](https://github.com/THUDM/slime)
- [slime 官方文档](https://thudm.github.io/slime/)
- [自定义 reward 指南](docs/zh/get_started/customization.md#--custom-rm-path)
- [保留的上游英文 README](README_origin.md)
- [保留的上游中文 README](README_zh_origin.md)

本项目沿用 slime 的 [Apache License 2.0](LICENSE)。
