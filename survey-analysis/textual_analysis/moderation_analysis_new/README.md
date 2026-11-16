# Moderation theory coding

对 pre-ML 和 post-ML theory 分别编码。一次 API 调用处理一位受访者的一项任务（race 或 gender）：pre 编码三个自选变量对；post 先将六个来源位置合并为 3–6 个唯一变量对，AI 编码后再映射回六个位置。

## 文件位置

- `prompts/`：pre/post 编码 prompt。
- `data/moderation_analysis_data.csv`：输入数据。
- `scripts/code_moderation_pre_ML.py`、`scripts/code_moderation_post_ML.py`：编码入口。
- `scripts/coding_schema.py`：共享字段校验与符号一致性计算。
- `scripts/fig_theory_mechanism_types_pre_post.py`：绘图及汇总表。
- `tests/`：离线测试。
- `outputs/`：编码生成的 `moderation_coding_pre_ML_v2.json`、`moderation_coding_post_ML_v2.json`，以及图、汇总表和 pair 明细。

Post 的 ML 输入继续读取 `survey-analysis/forecasts/ML_results_second_order_interactions.json`。

## 运行

在本文件所在文件夹运行：

```bash
python3 scripts/code_moderation_pre_ML.py
python3 scripts/code_moderation_post_ML.py
```

编码需要设置 `OPENAI_API_KEY`。两份脚本分别写入不同的结果文件，可以同时运行。每成功编码一段 theory 就保存，重跑时跳过结果文件中已有的 participant × task。

离线检查（不调用 API）：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/code_moderation_post_ML.py --dry-run
```

两阶段编码完成后绘图：

```bash
python3 scripts/fig_theory_mechanism_types_pre_post.py
```

脚本默认输入和输出路径根据文件位置解析，不依赖终端的工作目录。`--output` 等自定义相对路径仍相对于终端的工作目录。

## 编码与汇总

每个 pair 独立记录 `curvature`、`interaction_based` 和 `other_mechanism`，允许共存，不再输出主分类。Schema 标识为 `mechanism_presence_stages_v2`；旧 schema 的结果不会直接续用。

三项 `present` 仅允许 `yes/no`：`yes` 表示明确提供了该机制，`no` 表示未明确提供（包括表述含糊），不代表机制不可能存在。主图以并列柱展示 Curvature、Interaction-based、Other mechanism、No mechanism 四项比例，允许重叠，百分比不要求合计 100%。详细 CSV 仍保留八种组合和原始取值。`interaction_location` 和 `no_mechanism_reason` 仍可使用 `unclear`。独立字段 `discussed` 区分未讨论（`no`）与讨论过（`yes`）；讨论过但三项机制均为 `no` 表示没有编码出这三类机制。统计排除 `discussed=no` 的 pair；post 仍按无序变量对去重。

`discussed=yes` 且三项机制全部为 `no` 时，`no_mechanism_reason` 区分 `restatement` 和 `unclear`；其余情况（包括未讨论）为空字符串 `""`。旧结果中的该占位值在读取时兼容转换，不重新编码。该字段保存在编码 JSON 和用于绘图的 pair 明细 CSV 中（图表仍排除未讨论的 pair）。

编码模型不接收单独的 respondent predicted sign 或 ML sign，也无需复述它们；只根据 theory 判断 `mechanism_implied_sign`。原始符号由 Python 从输入补回结果并计算一致性。Theory 原文保持完整，其中自行提及的符号不会删除。

## 保存的符号字段

Pre 和 post 的每个 `coding.pairs` 对象只保存三项符号信息：

- `mechanism_implied_sign`：AI 根据 theory 判断，取 `positive/negative/underdetermined`。
- `reference_sign`：Python 从原始输入补入，取 `+/-`。Pre 使用受访者预测；post 的 `USER_SELECTED` 位置使用受访者预测，`ML` 位置使用 ML 估计。
- `sign_consistency`：Python 计算，取 `yes/no/underdetermined`。

后两项不交给模型输出，也不发送给模型。来源和排名仍可由 pre 的 `rank` 或 post 的 `pair_id` 识别。同一个 post 变量对只交给 AI 编码一次，映射回不同来源位置后，可以拥有不同的 reference sign 和 sign consistency，机制编码与 theory-implied sign 则一致。

不再额外保存重复的 `pairs_submitted`、`pairs_user_selected`、`pairs_ml_selected`、`pairs_to_code` 输入副本；完整变量对、原始符号和编码均在 `coding.pairs` 中，理论原文和输入文件来源仍保留。原始 CSV 和 ML 文件不作修改。

## Post 的去重与来源映射

Python 合并三个 respondent-selected 和三个 ML pair，按无序变量对去重，并按变量名排序，给模型提供 `pair_1`–`pair_N`（N=3–6）。模型只看到变量对、定义、Y 和 theory，不提供 ML 背景、来源、原始排名或单独的参考符号；理论原文不改写。

AI 返回 N 个 pair 的编码后，Python 根据特征组合恢复为六个原始 `pair_id`，各自补入 source、排名、reference sign 和 sign consistency。输出结构和绘图的来源去重逻辑保持兼容。`coding_protocol=neutral_unique_pairs_v1` 标记新的 post 编码流程；已有旧流程的 post 结果需另存新文件，避免混合续跑。Pre 不受该流程变更影响。

Pre/post 的 prompt 都逐项列出 pair 的变量及定义。Post 模板中的 `{{#PAIR_4}}`–`{{#PAIR_6}}` 是可选区块：Python 仅保留实际存在的 pair，区块标记不会发送给模型。

主图：`outputs/fig_theory_mechanism_presence_pre_post_v2.svg`；四项比例汇总：`fig_theory_mechanism_presence_pre_post_v2_summary.csv`。八组合汇总仍为 `fig_theory_mechanism_joint_type_pre_post_v2_summary.csv`。

## 使用更新后的 prompt 全量重跑

当前仅保留 v2 结果，编码和绘图默认均使用 v2。直接运行默认编码命令会跳过已有记录；全量重跑请通过 `--output` 指定新的文件，例如 `outputs/moderation_coding_pre_ML_rerun.json` 和 `outputs/moderation_coding_post_ML_rerun.json`。新的 curvature 定义允许直接发生在 Y 或其概率上的非线性解释，不要求中介 M。

## V2 的三种图

```bash
python3 scripts/fig_theory_mechanism_types_pre_post.py --version v2 --chart all
```

`--chart presence` 画四项平行比例，`--chart stacked` 画八种组合的 100% 堆叠图，`--chart change` 画 Post − Pre 的百分点变化图。分别保存为 `fig_theory_mechanism_presence_pre_post_v2`、`fig_theory_mechanism_joint_type_pre_post_v2`、`fig_theory_mechanism_change_pre_post_v2`（仅 SVG）。变化图的 CSV 同时保留两阶段的计数、分母、比例和未四舍五入的百分点差；它展示比例变化，不表示同一变量对的配对转移。
