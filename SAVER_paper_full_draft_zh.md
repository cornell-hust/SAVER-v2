# SAVER：面向无查询早期视频异常理解的搜索、告警与验证

## 摘要

视频异常理解正在从二值异常评分走向更具操作性的设定：系统不仅要判断视频中是否存在异常，还要决定异常在何时变得可采取行动，并用能够经受审视的视觉证据来支撑这一决策。然而，现有的视频异常检测与异常理解方法大多仍是被动的：它们要么在密集观测的片段上进行推理，要么是在相关内容已被观测到或由外部查询选出之后再提升答案质量。这就留下了长时监控视频中“无查询异常发现”的核心挑战：在有限观测预算下，模型必须自己决定去看哪里、当前获得的证据是否已经足够，以及何时发出告警。我们提出 **SAVER**，将早期视频异常理解建模为一种在策略闭环中包含显式告警决策与验证器反馈的主动证据获取过程。SAVER 结合了一个轻量级边界感知 scout、一个具有 `continue`、`soft_alert`、`hard_alert` 与 `declare_normal` 动作的主动搜索策略、**TriCEV**（一个对 `full`、`keep` 与 `drop` 三种窗口级证据视图进行对比的三视图反事实证据验证器），以及 **CGRPO**（一种用于在预算约束下稳定验证器感知搜索的保守型组相对裁剪策略优化方案）。在 TriCEV 中，当模型仅基于所选证据进行判断时，异常结论应当仍然成立；而当这些证据被移除后，异常判断应当显著下降。在 `soft_alert` 之后，如果当前证据支撑不足，系统可以继续搜索；在最终输出前，TriCEV 会对证据子集进行精炼，或在严格验证不满足时回退到 best-effort 子集。为评估这一设定，我们提出 **SAVER-Bench**，一个统一 MSAD、CUVA 与 ECVA 的时间优先 benchmark，包含异常区间、先兆区间、最早告警监督、证据时刻、反事实标注、关键对象、摘要、理由与问答。总体而言，SAVER 与 SAVER-Bench 将异常理解从对已观测片段的被动识别，重新刻画为预算约束下的无查询、验证器感知、早期决策问题。

## 1. 引言

经典视频异常检测（VAD）关注的是一个相对狭窄的问题：某一帧、某一片段或整段视频是否异常？围绕这一问题，学界已经通过重建、未来预测、弱监督以及近年来的开放词汇识别等方向取得了大量进展 [1-7]。但在实际部署中，一个有用的监控系统不能只给出异常分数。它还应当说明发生了什么、将事件定位到何时发生、尽可能早地发出预警，并用人类操作员能够检查的证据来解释这一预警。这一更广义的需求推动了任务从异常检测向 **视频异常理解**（VAU）的转变 [7-13]。

使这一转变变得困难的，不只是语义输出更丰富，还在于长时监控视频的序列性。相较于视频时长，异常内容通常极为稀疏，因此高成本的密集观测并不高效。更重要的是，早期预警本质上是一个在线决策问题：在事件尚未完全展开之前，系统必须判断到目前为止看到的证据是否已经足以支持介入。解释也不只是语言生成问题。只有当一段理由真正扎根于那些一旦被移除就会实质性改变异常判断、时间定位或告警决策的证据时，它才是可信的。这些约束表明，异常理解不应只围绕“预测什么”，还应围绕“看哪里”“何时停止”以及“所选证据是否真的足够”来建模。

这一视角使我们的目标问题与两条相近但并不相同的研究线索区分开来。其一，被动式的 LVLM 异常推理方法能够在已观测视频上提升语义输出质量，但它们并未将“观测本身”视为一个待学习的决策过程。其二，近期基于 RL 的异常推理方法，如 VAU-R1，虽然改进了推理行为，但依然没有将观测、告警时机与证据验证统一建模为同一个策略问题。在我们的设定中，并不存在用于定位异常的外部查询、答案目标或问题条件检索目标。异常可能出现在任意位置，也可能根本不存在。模型必须自主发现可疑证据、决定这些证据是否已经足以支撑告警，并在证据尚未得到充分验证时继续搜索。因此，SAVER 优化的是预算约束下 **如何搜索、如何验证、如何决策**，而不只是提升对已观测内容的下游推理质量。

基于此，我们主张：**早期视频异常理解是一个主动决策问题，而不是一个被动的事后推理问题。** SAVER，即 **Search, Alert, and Verify**，正是围绕这一主张构建。系统首先利用一个轻量级 scout 对整段视频进行摘要，生成时间 proposals 与异常先验。在这些先验条件下，主动策略决定应检查哪个 proposal、使用何种时间尺度、何种模态，以及是继续观察、发出试探性告警、提交最终告警，还是宣布视频正常。搜索轨迹不断更新 belief state，以支撑异常识别、时间定位、先兆估计、最早告警预测以及证据选择。随后，**TriCEV** 这个三视图反事实证据验证器会在 `full`、`keep` 与 `drop` 三种窗口级证据视图下检验证据是否充分，从而使系统能够在 `soft_alert` 证据不足时继续搜索，或在生成结构化输出与语言输出之前对最终证据子集进行精炼。

为评估这一设定，我们提出 **SAVER-Bench**，一个时间优先的异常理解 benchmark，将 MSAD、CUVA 与 ECVA 统一到同一个 schema 与 evaluator 下。SAVER-Bench 恰好暴露了被动式 VAD benchmark 往往掩盖的性质：先兆区间、最早告警目标、搜索得到的证据时刻、反事实标注以及与搜索相关的评估指标。因此，本文的中心论点可以概括为：**SAVER 将异常理解从对已观测片段的被动识别，转变为预算约束下无查询、验证器感知的证据获取过程。** 具体而言，我们有三点贡献。第一，我们将主动式早期异常理解形式化为一个无查询决策问题，其中观测、告警时机与证据可信性都是一等目标。第二，我们提出 SAVER 这一统一的 `Scout -> Search -> Alert -> Verify -> Refine -> Generate` 框架，其中包含 **TriCEV** 这一作用于已搜索时间证据子集的三视图反事实证据验证器，以及 **CGRPO** 这一任务特定的训练方案，用于通过组相对轨迹比较、PPO 风格裁剪与启发式 continuation 来稳定预算约束下的验证器感知搜索。第三，我们提出 SAVER-Bench 及其匹配的评测协议，以评估时间定位、最早告警、证据可信性、搜索效率与辅助语言质量。

## 2. 相关工作

### 2.1 经典与弱监督视频异常检测

早期 VAD 方法主要将异常建模为对已学习正常规律的偏离，通常通过重建或未来预测来识别偏离正常视频动态的事件 [1, 2]。随后，弱监督方法利用多实例学习、排序目标与噪声鲁棒训练，在长时监控视频中的时间定位能力上取得了显著提升 [3-6]。这些方向在异常判别与粗粒度定位上推动了领域发展，但它们共享一个被动假设：模型对预先给定的视频流进行打分，而不是主动决定下一步去检查哪些证据、系统何时应当告警，或者所选证据是否足以支撑这一告警。

### 2.2 开放词汇与 LVLM 视频异常理解

开放词汇与大模型方法将异常分析扩展到固定类别体系与标量异常分数之外。Open-Vocabulary Video Anomaly Detection [7] 使异常识别可以覆盖更丰富的语义类别。Harnessing Large Language Models for Training-free Video Anomaly Detection [18] 与 Follow the Rules [9] 进一步表明，语言模型监督能够改进监控视频上的异常推理与自由形式解释。Towards Surveillance Video-and-Language Understanding [19] 则在数据集层面推进了这一方向，将监控理解表述为更丰富的视频语言问题。Anomize [12] 则沿着开放词汇路线进一步强化了语义对齐能力。这些工作之所以重要，是因为它们推动了异常理解向更丰富语义输出发展；但它们大多仍然默认相关内容已经被观测到，或已经被密集采样。相比之下，SAVER 将 **观测本身** 视为学习问题的一部分。

### 2.3 可解释与证据扎根的异常推理

CUVA 通过询问发生了什么、为什么发生以及异常如何展开，迈出了从二值检测走向结构化异常理解的重要一步 [8]。VERA 通过视觉语言学习将异常证据 verbalize 出来 [10]，而 Holmes-VAU 则研究了跨不同粒度的长时异常理解 [11]。这些工作都直接启发了我们的设定，但它们仍未解决一个与部署密切相关的问题：解释质量不仅应取决于对已观测视频进行语言化表达的能力，还应取决于 **系统在预算约束下主动获取了哪些证据**，以及这些证据在反事实检验下是否仍然能够支撑结论。SAVER 正是通过 **TriCEV** 来填补这一缺口：它显式对比已搜索时间证据子集在三种证据视图下的表现，而不是仅仅把解释视作对固定观测上下文的语言生成。

### 2.4 面向长视频理解的主动搜索与强化学习

序列决策天然适用于长视频分析。Holmes-VAU 已经强调，自适应时间搜索是长时异常理解中的关键组成部分 [11]。近期如 VAU-R1 这样的 RL 异常推理工作，则通过强化微调改进了异常推理行为 [13]。PPO 风格的裁剪优化依然是这类顺序策略的实用基础 [15]。SAVER 将这一方向进一步扩展到更具操作性的设定：策略必须主动获取证据、发出显式告警、在验证不足的 `soft_alert` 之后继续搜索，并在未发现异常时宣布正常。因此，这里的强化学习并不是用来优化答案风格，而是用来优化预算约束下的搜索效率与决策时机。特别地，SAVER 并不采用纯语言风格的组相对目标。其训练中的 **CGRPO** 是一种任务特定方案：它比较同一视频上的多条搜索轨迹，在轨迹层面保留 PPO 风格裁剪，并在 learned TriCEV 仍处于对齐过程中时维持启发式 continuation。

## 3. 问题设定与 SAVER-Bench

### 3.1 任务定义

我们考虑一段长时监控视频

\[
V = \{f_t\}_{t=1}^{T},
\]

其被划分为 clip 单元

\[
\mathcal{C} = \{c_i\}_{i=1}^{N}.
\]

模型在有限观测预算 \(B\) 下与视频交互。在第 \(k \leq B\) 步，模型选择一个 proposal 进行检查，选择一个时间尺度与模态，然后决定是继续、发出试探性预警、提交最终告警，还是宣布视频正常。因此，系统输出既包括结构化预测

\[
\hat{\mathcal{Y}} =
(\hat{y}^{exist}, \hat{y}^{cat}, \hat{s}, \hat{I}, \hat{P}, \hat{t}_{alert}, \hat{E}, \hat{c},
\hat{g}^{sum}, \hat{g}^{rat}, \hat{g}^{cf}, \hat{Q}),
\]

也包括搜索轨迹

\[
\hat{\tau} = \{(a_k, W_k, u_k)\}_{k=1}^{K}, \qquad K \leq B.
\]

其中，\(\hat{y}^{exist}\) 表示异常是否存在，\(\hat{y}^{cat}\) 表示异常类别，\(\hat{s}\) 表示严重程度，\(\hat{I}\) 表示异常区间，\(\hat{P}\) 表示先兆区间，\(\hat{t}_{alert}\) 表示最早可行动告警时刻，\(\hat{E}\) 表示证据子集，\(\hat{c}\) 表示反事实类型，\(\hat{g}^{sum}\)、\(\hat{g}^{rat}\) 与 \(\hat{g}^{cf}\) 分别表示生成的摘要、理由与反事实解释，\(\hat{Q}\) 表示问答输出，而 \(u_k\) 表示触发验证时的 TriCEV 状态。该设定的关键不只是结构化预测本身，而是 **部分观测条件下的无查询决策**：模型必须发现异常是否存在、获取足以支撑告警的证据，并决定何时告警才算可行动。

### 3.2 为什么现有 benchmark 不足

大多数经典 VAD benchmark 主要面向异常评分或粗粒度时间定位。即便是近期的异常理解资源，也主要在相关内容已被观测到之后评估语义解释能力 [8-13, 19]。这使三个与部署密切相关的问题仍缺乏充分监督：模型能否高效搜索而不是密集观测？它能否尽早发出预警而不只是事后定位？它能否用在反事实检验下仍然关键的证据来支撑自身决策？这些问题需要显式的先兆区间、最早告警监督、证据时刻以及搜索感知评估指标。SAVER-Bench 正是围绕这一缺口设计的。

### 3.3 Benchmark Schema 与数据集组成

SAVER-Bench 中的每条样本都遵循一个面向论文表述的 schema，其核心是主动异常理解所必需的信息。在时间层面，样本可包含异常区间、先兆区间、最早告警帧以及证据时刻。在语义层面，样本包含异常类别、严重程度、反事实类型与文本，以及关键对象。在语言层面，benchmark 提供摘要、理由与问答监督。

当前发布版本将 MSAD [16]、CUVA [8] 与 ECVA 统一到训练代码所采用的同一 schema 与划分逻辑中。完成 split 解析后，SAVER-Bench 共包含 3,880 段视频。

| 来源 | 视频数 | 异常 | 正常 | 训练 | 测试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSAD | 720 | 240 | 480 | 480 | 240 |
| CUVA | 986 | 986 | 0 | 786 | 200 |
| ECVA | 2,174 | 2,174 | 0 | 1,536 | 638 |
| 总计 | 3,880 | 3,400 | 480 | 2,802 | 1,078 |

这一统一版本还存在一个需要明确说明的现实限制。MSAD 同时包含正常与异常视频，而当前 CUVA 与 ECVA 的转换版本仅包含异常样本。因此，存在性 AP、误报率与 hard-normal 误报率等指标，在 MSAD 或包含正常视频的混合来源设定上更具信息量。

### 3.4 评估协议

评估器围绕四类指标组织。**主指标** 包括存在性 AP、类别 Macro-F1、时间 mIoU、时间 R@1@0.5、先兆 mIoU、先兆 R@1@0.5、告警效用、过早告警率、误报率、证据 F1@3 以及反事实类型准确率。**辅助指标** 包括 evidence precision@3、evidence recall@3 以及 hard-normal 误报率。**搜索效率指标** 包括平均检查 clip 比例、平均搜索步数与平均时延。**辅助语言指标** 包括严重程度 MAE、摘要 ROUGE-L、理由 ROUGE-L、理由精确匹配、反事实文本 ROUGE-L、反事实文本精确匹配、问答答案 ROUGE-L 以及问答答案精确匹配。该协议的设计目标是使本文主张可被直接测量：一个好的系统不仅应当“答对”，还应当更早、更高效、且更有证据支撑。

## 4. 方法

### 4.1 总览

SAVER 沿着一条单一因果链展开：

\[
\text{Scout} \rightarrow \text{Search} \rightarrow \text{Alert} \rightarrow \text{Verify} \rightarrow \text{Refine} \rightarrow \text{Generate}.
\]

这一分解不是写作便利，而是方法本身的核心。Scout 用于解决全视频成本问题，生成 proposal memory 与异常先验。Search 策略解决部分观测问题，决定下一步查看什么。显式告警使“可行动性”成为决策过程的一部分，而不只是一个事后阈值。Verify 检验证据是否真的充分且必要。Refine 则保证最终结构化输出与语言输出依赖的是最可辩护的证据子集，而不是简单依赖原始访问上下文。

### 4.2 边界感知 Scout

在高成本检查开始之前，SAVER 先对 clip 时间线做一次低成本扫描。对于每个 clip，scout 提取轻量级外观统计、运动幅值与相对时间位置。这些线索用于估计边界与 eventness，再据此将初始片段合并为事件 proposals。每个 proposal 同时获得一个手工构造的边界感知特征，以及一个在 scouting prompt 下使用 Qwen2.5-VL [14] 对少量 scout 帧编码得到的语义特征。随后，一个时间编码器生成 proposal memory

\[
M^0 = \{m_j^0\}_{j=1}^{N_p},
\]

以及 proposal 级 prior logits 与异常先验概率 \(q_j\)。因此，下游策略不是从空白状态开始，而是从一个低成本的全局可疑度图出发。

### 4.3 POMDP 形式化与主动搜索

我们将主动异常理解建模为一个部分可观测马尔可夫决策过程

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, \gamma).
\]

在第 \(k\) 步，agent 选择

\[
a_k = (j_k, s_k, m_k, d_k),
\]

其中 \(j_k\) 是 proposal 索引，\(s_k\) 是时间尺度，\(m_k \in \{\texttt{rgb}, \texttt{motion}, \texttt{rgb+motion}\}\) 是观测模态，而 \(d_k \in \{\texttt{continue}, \texttt{soft\_alert}, \texttt{hard\_alert}, \texttt{declare\_normal}\}\) 是决策动作。策略可分解为

\[
\pi_\theta(a_k \mid h_k, M^0) =
\pi_\theta^{idx}(j_k \mid h_k, M^0)\,
\pi_\theta^{scale}(s_k \mid h_k)\,
\pi_\theta^{mod}(m_k \mid h_k)\,
\pi_\theta^{dec}(d_k \mid h_k),
\]

其中 \(h_k\) 表示当前 belief state。Proposal 索引 logits 会被 scout priors 所偏置，从而将低成本 scouting 与高成本 inspection 耦合到同一个决策回路中。

将 \(m_k\) 纳入动作空间对问题定义十分重要。在 SAVER 中，一次 inspection step 本质上是一次证据获取操作，而不只是一个时间指针：在决定去哪里看、看多大范围之后，策略还必须决定向 backbone 请求哪一类证据。固定使用 `rgb+motion` 虽然是一个合理 baseline，但它会让每一步都退化为相同的“总是最丰富”的观测算子，从而抹去策略对证据类型进行自适应控制的能力。在部分观测条件下，先兆判断往往更依赖外观与场景上下文，而决定性触发时刻则更依赖短时动态。显式建模模态选择，因而保留了 POMDP 的本意：策略控制的不只是在哪里、什么时候看，还包括获取什么证据。

### 4.4 局部检查与 Belief 更新

给定动作 \(a_k\)，SAVER 会根据所选时间尺度扩展目标 proposal，并构造一个 inspection window

\[
W_k = [j_k - \Delta(s_k),\ j_k + \Delta(s_k)].
\]

模型从该窗口中采样少量 RGB 帧、运动图像或二者的组合，并在 inspection prompt 下使用 Qwen2.5-VL 对其编码。得到的 embedding \(z_k\) 通过如下递归更新 belief：

\[
h_k = f_\theta(h_{k-1}, z_k).
\]

这一设计更偏务实，而不是单独的创新点。在当前实现中，`rgb+motion` 是将 RGB 帧与轻量级运动图像拼接，`motion` 则用帧差近似而非稠密光流实现。因此，我们将固定 `rgb+motion` 视为 ablation 中的重要对照，而不把它视作与策略模态选择等价。模态选择的作用更窄也更操作化：**在部分观测条件下，agent 不仅应控制看哪里、看多宽，还应控制从该窗口取回什么类型的证据。**

与此同时，SAVER 维护一个显式的已访问窗口集合

\[
\mathcal{W}_k = \mathcal{W}_{k-1} \cup \{W_k\},
\]

因为证据选择、验证以及最终生成都必须与实际搜索轨迹本身绑定，而不能只依赖最后一个隐藏状态。

### 4.5 结构化理解与显式告警

在 \((M^0, h_k, \mathcal{W}_k)\) 条件下，understanding module 预测异常存在性、异常类别、严重程度、异常区间、先兆区间、最早告警指针、已访问窗口上的证据 logits、证据角色以及反事实类型。一个关键约束是：证据只能从 **已访问窗口** 中选取。这样可以防止模型悄悄用从未真正查看过的片段为自己的决策辩护。

告警被建模为显式决策，而不是时间定位的副产物。当策略输出 `soft_alert` 或 `hard_alert` 时，SAVER 会检查当前结构化预测是否满足一个告警置信度准则。第一个满足该准则的告警会成为 **policy alert**，其解码出的指针定义当前的最早告警估计。随后，这一告警状态会重新注入 belief dynamics，使告警后的行为可以不同于告警前的探索。`soft_alert` 与 `hard_alert` 的区分是有意设计的：`soft_alert` 非终止，它允许系统先发出试探性告警，同时保留继续搜索的选项；`hard_alert` 则是终止动作，会在输出提交前触发最终验证。

### 4.6 TriCEV：三视图反事实证据验证

设 \(\hat{E} \subseteq \mathcal{W}_K\) 为停止时刻选出的证据子集。SAVER 构造三种反事实视图：

\[
\mathcal{W}^{full} = \mathcal{W}_K, \qquad
\mathcal{W}^{keep} = \hat{E}, \qquad
\mathcal{W}^{drop} = \mathcal{W}_K \setminus \hat{E},
\]

并在三种视图上重新计算结构化支持。我们将这一步具体实现为 **TriCEV**（Tri-View Counterfactual Evidence Verifier）。TriCEV 分别编码三个视图，并利用三种编码及其成对差分，同时预测视图得分与四分类证据状态。TriCEV 输出视图得分

\[
s_{full},\ s_{keep},\ s_{drop},
\]

以及一个离散状态

\[
u \in \{\texttt{complete}, \texttt{incomplete}, \texttt{redundant}, \texttt{misaligned}\}.
\]

直观而言，`complete` 表示仅保留证据的视图仍然有强支持，而去掉这些证据后的视图发生明显塌陷；`redundant` 表示两者都依然强；`incomplete` 表示所选证据本身不足；`misaligned` 表示所选证据实际上并不支撑当前预测。需要强调的是，TriCEV 是一个 **窗口级时间证据验证器**，而不是对象遮挡式的强因果干预引擎。因此，本文主张的是时间证据层面的反事实验证，而不是对象级强因果编辑。

### 4.7 TriCEV 引导的 continuation、refinement 与 best-effort fallback

验证并不是一个纯诊断性的后处理，而是策略闭环中的一部分。在推理时，如果某次 `soft_alert` 之后 TriCEV 给出 `incomplete` 或 `misaligned`，且预算仍然充足，则 SAVER 会继续搜索。当验证结果令人满意时，SAVER 会移除那些被排除后对 TriCEV 支持几乎没有影响的窗口，从而得到

\[
\hat{E}_{verified} \subseteq \hat{E}.
\]

如果严格验证不满足，SAVER 会退回到一个由 TriCEV 评分的 best-effort 子集

\[
\hat{E}_{best} = \arg\max_{\varnothing \neq E' \subseteq \hat{E}} g_{ver}(E').
\]

由于证据集合的基数被刻意保持较小，因此这一搜索仍然是可 tractable 的。最终结构化输出会基于 verified 或 best-effort 子集重新计算，而不是直接从原始搜索轨迹中复制。这一步对于本文主张至关重要：解释质量取决于模型主动搜索得到的证据在验证之后是否仍然站得住脚。

这里有一个对训练的 code-faithful 表述很重要的实现细节。尽管推理阶段默认使用 learned TriCEV status，但在训练中，系统并不会让 learned TriCEV 直接决定一次 `soft_alert` 应继续还是终止。相反，continuation 由一个基于 `full`、`keep` 与 `drop` 三视图相对支持关系计算得到的 **启发式验证状态** 所驱动。这一选择是刻意的。在 SAVER 中，一次错误的继续搜索主要只是多消耗一些预算；而一次错误的过早停止则可能使策略永远看不到决定性证据，从而永久损害该轨迹上的告警时机与证据质量。由于 TriCEV 最强的显式监督出现在后续阶段，因此采用启发式 continuation 能在策略学习搜索时保持更保守、方差更低的分支控制。与此同时，训练中还加入了辅助 TriCEV loss，使 learned TriCEV 的状态 logits 与视图得分对齐到相同的启发式支持目标，从而在不直接扰乱 rollout 控制的前提下优化验证器。

### 4.8 Verified-first 语言生成

在 refinement 之后，SAVER 构造用于摘要生成、理由生成、反事实解释与问答的 prompts。这些 prompts 以异常类别、严重程度、时间区间、证据角色、场景与关键对象等结构化预测为条件。视觉上下文遵循一个严格优先级：优先 verified evidence windows，其次是 selected evidence windows，再其次是 visited windows，最后才是 top-prior scout proposals。因此，生成策略是 **verified-first with fallback**，而不是严格的 verified-only generation。系统会强烈偏好经过 TriCEV 精炼的上下文，但保留 fallback 路径以避免空上下文失败。

### 4.9 四阶段训练

SAVER 采用四阶段训练。**Stage A** 执行 teacher-guided understanding pretraining：将 oracle trajectories 通过与推理时完全相同的 scout、inspection 与 belief-update 模块进行 replay。这样可以为异常存在性、类别、严重程度、区间、告警序列、最早告警指针、证据选择、证据角色、scout priors、边界质量、先兆预测与反事实类型等监督建立正确的部分观测上下文。**Stage B** 通过 behavior cloning 初始化搜索策略：

\[
\mathcal{L}_{BC} = - \sum_k \log \pi_\theta(a_k^{oracle} \mid h_k, M^0).
\]

在实现上，behavior cloning 同时监督 proposal 选择、时间尺度、模态选择与决策类型，在 Stage C 用任务奖励进一步优化之前，为观测控制提供稳定初始化。

**Stage C** 执行我们称为 **Conservative Group-Relative Proximal Optimization（CGRPO）** 的训练。对于每个视频，模型会采样一小组轨迹，计算证据命中、新颖性与重复访问控制等逐步奖励，并加入一个综合分类质量、时间定位、告警质量、效率、验证质量与反事实正确性的终止奖励。回报会在同一组 rollout 内做归一化，从而在没有单独 critic 的情况下得到组相对优势：

\[
A_g = \frac{R_g - \mu(R)}{\sigma(R) + \epsilon}.
\]

采样轨迹随后会在 teacher-forced 状态转移下 replay，并使用轨迹对数概率上的 PPO 风格裁剪比率来更新策略：

\[
\mathcal{L}_{\mathrm{CGRPO}}
=
-\mathbb{E}_{g}\left[\min\left(\rho_g A_g,\ \operatorname{clip}(\rho_g, 1-\delta, 1+\delta)A_g\right)\right]
- \beta \mathcal{H}
+ \lambda_{\mathrm{aux}} \mathcal{L}_{\mathrm{ver}}^{aux}
\]

其中 \(\rho_g\) 为当前轨迹概率与 rollout 轨迹概率之间的比率。其保守性体现在两个方面。第一，在 rollout 中，`soft_alert` 之后的 continuation 由启发式状态控制，而不是由 learned TriCEV status 直接控制，因为此时 TriCEV 尚未完全训练好，错误的提前停止比多搜索几步更具破坏性。第二，轨迹层面的 PPO 风格裁剪被保留下来，以限制高成本分支式视频搜索问题中的更新方差。更具体地说，CGRPO 是一种 **没有单独 critic、也没有 reference model 的任务特定分组裁剪式 RL 阶段**。这一点对定位十分重要：SAVER 并不是用 RL 去优化答案风格或 chain-of-thought 形式，而是用 RL 来优化预算约束下的搜索、证据充分性与告警时机。

**Stage D** 固定轨迹收集为确定性的 learned-policy rollouts，并利用从 ground-truth evidence moments 派生出的 complete、incomplete、redundant 与 misaligned 证据掩码来训练 TriCEV，同时在 verified-first 上下文策略下优化摘要、理由、反事实与问答生成。这一阶段的作用不是给一个已经完成的 detector 事后贴上解释，而是让基于 TriCEV 的证据精炼与语言输出与实际搜索轨迹对齐。

## 5. 实验

当大规模对比实验仍在刷新时，我们不会报告暂时不稳定的数值，而是将相应单元格留空。实验结构与比较逻辑是固定的。实验围绕四个问题展开：SAVER 是否能在 MSAD 上提升面向部署的异常理解；它是否能在多来源设定下提升统一异常理解；`Search -> Alert -> Verify` 分解中的哪些部分最关键；以及相较于被动观测，主动搜索能带来怎样的成本质量权衡。

### 5.1 实验设置

除非另有说明，实验均使用 Qwen2.5-VL-7B [14] 作为多模态 backbone。我们的默认设定采用 LoRA，rank 为 16，scaling 为 32，dropout 为 0.05，使用 bf16 权重与梯度检查点。搜索配置使用 32 帧 clips、每段视频最多 48 个 event proposals、半径为 \(\{0, 1, 3\}\) 的三种时间尺度、三种模态（`rgb`、`motion`、`rgb+motion`）、训练时 8 个搜索步、评估时 10 个搜索步，以及 top-3 evidence windows。我们报告三种设定：用于异常与正常区分及误报分析的 **MSAD-only** 设定；用于语义与时间理解的 **all-source anomaly-understanding** 设定；以及跨 MSAD、CUVA 与 ECVA 的 **per-source** 结果拆分。

### 5.2 比较协议

我们与三类 baseline 进行比较。第一类是被动异常检测器，包括弱监督 VAD 与开放词汇 VAD 系统，用于检验主动搜索相较于密集观测或固定观测是否具有更好的质量成本权衡。第二类是异常理解 baseline，包括 CUVA 风格的结构化理解、VERA、Holmes-VAU，以及在适配可行时的其他被动 LVLM 流水线。第三类是 SAVER 自身的 ablations，用于隔离主动搜索、显式告警、TriCEV 引导 continuation、证据精炼以及 Stage C 中 **CGRPO** 设计的价值。由于若干已发表 baseline 并非为无查询主动推理设计，我们会仔细说明适配细节，并尽可能在同一 schema 与同一指标定义下评估所有方法。

### 5.3 MSAD 上的主要结果

MSAD 是检验异常与正常区分以及告警风险的主要测试平台，因为它同时包含异常与正常视频。因此，关键问题不仅是 SAVER 是否能准确检测异常，更是它能否比被动 baseline **更早** 地做出判断，并使用 **支撑性更强的证据**。

| 方法 | 存在性 AP | 时间 mIoU | 时间 R@1@0.5 | 告警效用 | 过早告警率 | 误报率 | 证据 F1@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 弱监督 VAD baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 开放词汇 VAD baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 可解释异常 baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER w/o Stage C RL | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.4 统一异常理解结果

All-source 设定评估的是：主动搜索与证据验证是否能在存在性判断之外，提升语义与时间层面的异常理解。核心问题是，无查询证据获取是否能带来更好的类别预测、更好的时间定位、更好的先兆估计以及更可信的证据选择。

| 方法 | 类别 Macro-F1 | 时间 mIoU | 先兆 mIoU | 告警效用 | 证据 F1@3 | 反事实类型准确率 | 摘要 ROUGE-L | 理由 ROUGE-L | QA ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 被动 LVLM baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 开放词汇异常 baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 可解释异常 baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.5 按来源拆分的结果

由于 MSAD、CUVA 与 ECVA 在标注风格、异常 taxonomy 以及正常异常比例上均有差异，因此必须进行 per-source reporting。该表旨在检验 SAVER 的收益是否在不同来源上保持稳定，而不是仅仅依赖于某一个数据集。

| 来源 | 类别 Macro-F1 | 时间 mIoU | 先兆 mIoU | 告警效用 | 证据 F1@3 | 反事实类型准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSAD | TBD | TBD | TBD | TBD | TBD | TBD |
| CUVA | TBD | TBD | TBD | TBD | TBD | TBD |
| ECVA | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.6 消融实验

消融实验的目标不是调外围超参数，而是隔离本文核心主张。具体而言，它测试性能增益究竟来自主动搜索、TriCEV 引导 continuation、证据精炼与强化学习，还是仅仅来自 prompt engineering 或语言生成。一项重要对照是：将 learned modality choice 替换为每一步都固定使用 `rgb+motion`。它直接检验模态是否应当被视为动作空间的一部分，还是系统在总是使用最丰富观测时也表现近似。另一项重要对照是：用 learned-TriCEV continuation 替代启发式 continuation，以检验保守的 **CGRPO** 设计是否真的是稳定 TriCEV-aware search 所必需的，还是说即使在一个部分训练好的 TriCEV 已经进入 branching loop 的情况下，该策略也能同样稳定地训练。

| 变体 | 时间 mIoU | 告警效用 | 证据 F1@3 | 平均步数 | 平均检查 clip 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 SAVER | TBD | TBD | TBD | TBD | TBD |
| 每步固定 `rgb+motion` | TBD | TBD | TBD | TBD | TBD |
| 去掉主动搜索（被动或仅 top-prior） | TBD | TBD | TBD | TBD | TBD |
| Stage C 中用 learned-TriCEV continuation 替代启发式 continuation | TBD | TBD | TBD | TBD | TBD |
| 去掉 TriCEV-guided continuation | TBD | TBD | TBD | TBD | TBD |
| 去掉 verified-evidence refinement | TBD | TBD | TBD | TBD | TBD |
| 去掉 Stage C RL | TBD | TBD | TBD | TBD | TBD |
| 用 visited windows 而不是 verified-first context 做语言生成 | TBD | TBD | TBD | TBD | TBD |

### 5.7 搜索效率与定性分析

由于 SAVER 的动机就是预算约束下的观测，因此效率必须与决策质量一起报告，而不应被放在单独的工程附录中。最重要的搜索效率指标包括平均检查 clip 比例、平均搜索步数与平均时延，并需与时间定位与告警效用联合解释。

| 方法 | 平均检查 clip 比例 | 平均搜索步数 | 平均时延 | 时间 mIoU | 告警效用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 被动 baseline | TBD | TBD | TBD | TBD | TBD |
| SAVER | TBD | TBD | TBD | TBD | TBD |

定性分析围绕三类案例组织。第一类展示成功的早期预警轨迹：SAVER 发出 `soft_alert`，在 TriCEV 反馈后继续搜索，最终以一个精炼后的证据子集完成提交。第二类展示 TriCEV 驱动的证据纠错：原始选中的证据可能是冗余的或不完整的，而 TriCEV 会给出一个更紧凑或更有支撑力的子集。第三类展示失败案例，如告警过晚、证据错误或反事实错位。之所以要强调这些案例，是因为本文的核心主张关注的是可行动性与证据可信性，而不是最终文本表述是否流畅。

### 5.8 局限性

需要在最终实验中重点分析的局限包括：当前驱较弱或存在歧义时导致的告警延迟；在多个相关窗口都看似合理的情况下，系统未能保留真正决定性证据；以及当生成从 verified context 回退到更广 visited context 时，理由与反事实文本出现语义漂移。还必须明确指出，TriCEV 作用于时间证据子集，而不是显式对象级干预。这些局限并不会否定该框架，而是定义了主动异常理解目前仍然困难的真实边界。

## 6. 结论

本文提出了 SAVER，一个面向无查询早期视频异常理解的 **Search, Alert, and Verify** 框架。不同于在已观测片段上进行推理的被动式异常方法，SAVER 将观测本身视作预算约束下的待学习决策问题。轻量级 scout 提出可疑时间区域，主动策略决定检查什么以及何时告警，**TriCEV** 在时间证据窗口层面检验所选证据是否充分且必要，**CGRPO** 则在 TriCEV 尚未完全对齐之前稳定 TriCEV-aware search 的优化。最终预测基于 verified 或 best-effort 证据进行精炼后，再用于扎根语言生成。结合 SAVER-Bench，这一形式化为从被动异常评分走向主动、早期、证据扎根的长视频异常理解提供了一条具体路径。

## 参考文献

[1] Hasan M, Choi J, Neumann J, Roy-Chowdhury AK, and Davis LS. Learning temporal regularity in video sequences. In CVPR, 2016.

[2] Liu W, Luo W, Lian D, and Gao S. Future frame prediction for anomaly detection: A new baseline. In CVPR, 2018.

[3] Sultani W, Chen C, and Shah M. Real-world anomaly detection in surveillance videos. In CVPR, 2018.

[4] Zhong JX, Liu N, Li F, Ren W, Gao S, and Sebe N. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for video anomaly detection. In CVPR, 2019.

[5] Feng J, Kumar A, Schadenberg B, and Wachsmuth S. MIST: Multiple instance self-training framework for video anomaly detection. In CVPR, 2021.

[6] Tian Y, Zhao G, Wang Y, and Loy CC. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In ICCV, 2021.

[7] Wu P, Wang X, Cao Y, and Ding S. Open-vocabulary video anomaly detection. In CVPR, 2024.

[8] Du H, et al. Uncovering What, Why and How: A comprehensive benchmark for causation understanding of video anomaly. In CVPR, 2024.

[9] Yang Y, et al. Follow the Rules: Reasoning for video anomaly detection with large language models. In ECCV, 2024.

[10] Ye M, et al. VERA: Explainable video anomaly detection via verbalized learning of vision-language models. In CVPR, 2025.

[11] Zhang H, et al. Holmes-VAU: Towards long-term video anomaly understanding at any granularity. In CVPR, 2025.

[12] Li F, et al. Anomize: Better open vocabulary video anomaly detection. In CVPR, 2025.

[13] Zhu L, et al. VAU-R1: Advancing video anomaly understanding via reinforcement fine-tuning. arXiv preprint arXiv:2505.23504, 2025.

[14] Bai S, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[15] Schulman J, Wolski F, Dhariwal P, Radford A, and Klimov O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[16] Zhu L, et al. Advancing video anomaly detection: A concise review and a new dataset. In NeurIPS Datasets and Benchmarks Track, 2024.

[17] Simonyan K and Zisserman A. Two-stream convolutional networks for action recognition in videos. In NeurIPS, 2014.

[18] Zanella L, Menapace W, Mancini M, Wang Y, and Ricci E. Harnessing large language models for training-free video anomaly detection. In CVPR, 2024.

[19] Yuan T, et al. Towards surveillance video-and-language understanding: New dataset, baselines and challenges. In CVPR, 2024.

## 供投稿准备的附录包

这一部分刻意采用面向论文而非面向仓库的写法。它固定了图、表与补充材料的内容逻辑，以便后续迁移到正式模板时无需改变叙事主线。

### A. 图示包

#### 图 1. Overall Framework

目的：展示 SAVER 是一个主动异常理解系统，而不是一个被动 VAD pipeline。

需要包含的内容：
- 输入长视频时间线
- 边界感知 scout 与 proposal memory
- 包含 proposal selection、scale、modality 与 decision action 的 search loop
- `soft_alert` 与 `hard_alert` 的显式区分
- 带有 `full`、`keep` 与 `drop` 证据视图的 **TriCEV**
- evidence refinement 或 best-effort fallback
- 最终结构化输出与语言输出

建议图注：*SAVER 总览。一个轻量级 scout 提出可疑时间区域，主动策略在预算约束下执行搜索并发出 soft 或 hard 告警，随后 **TriCEV** 这一三视图反事实证据验证器检查所选证据是否充分且必要，最后再生成结构化输出与语言输出。*

#### 图 2. SAVER-Bench 任务与标注 Schema

目的：可视化解释为什么 SAVER-Bench 是一个时间优先 benchmark。

需要包含的内容：
- 一条水平视频时间线
- 异常区间
- 先兆区间
- 最早告警帧
- 带语义角色的证据时刻
- 右侧面板展示 severity、counterfactual、key objects、summary、rationale 与 QA

建议图注：*SAVER-Bench 围绕主动异常理解中的时间监督组织，包括异常区间、先兆区间、最早告警帧、证据时刻以及扎根语言标注。*

#### 图 3. 定性搜索轨迹

目的：展示主动搜索与被动观测的差异。

需要包含的内容：
- 来自同一视频的 4 至 6 个搜索步骤
- 每一步的 selected proposal 与 inspected window
- 当前决策动作（`continue`、`soft_alert`、`hard_alert`）
- 在适用时 soft alert 之后的 TriCEV 状态
- 最终 verified 或 best-effort evidence subset

建议图注：*SAVER 的一个搜索轨迹示例。模型先发出 soft alert，在 TriCEV 反馈后继续搜索，直到证据子集变得足够有支撑性时才最终提交。*

#### 图 4. TriCEV 与 Best-Effort Fallback 案例

目的：展示验证会改变最终输出，而不是仅仅给它打分。

需要包含的内容：
- 原始选中的证据窗口
- TriCEV 状态
- 精炼后的或 best-effort 的证据子集
- refinement 之后的最终区间、告警与理由

建议图注：*TriCEV 能够将一个原始证据子集精炼为更紧凑的 verified 或 best-effort 子集，并因此改变最终预测与解释。*

### B. 表格包

#### 表 1. MSAD 上的主要比较

核心信息：SAVER 应当在异常识别、早期告警与证据质量上联合评估。

列：
- 存在性 AP
- 时间 mIoU
- 时间 R@1@0.5
- 告警效用
- 过早告警率
- 误报率
- 证据 F1@3

#### 表 2. 统一异常理解结果

核心信息：主动搜索与证据验证不应只提升异常检测，还应提升语义与时间理解。

列：
- 类别 Macro-F1
- 时间 mIoU
- 先兆 mIoU
- 告警效用
- 证据 F1@3
- 反事实类型准确率
- 摘要 ROUGE-L
- 理由 ROUGE-L
- QA ROUGE-L

#### 表 3. 按来源拆分结果

核心信息：揭示 MSAD、CUVA 与 ECVA 在领域与标签分布上的差异。

#### 表 4. 消融实验

核心信息：检验搜索、告警、验证与精炼是否都对核心主张有实质贡献。

#### 表 5. 搜索效率分析

核心信息：展示 SAVER 相对于被动观测是否具备更优的成本质量权衡。

### C. 附录骨架

在正式模板中可准备如下附录部分：
- A. 完整训练细节与超参数
- B. 数据集 schema 与标注示例
- C. scout、inspection、rationale、summary、counterfactual 与 QA 的 prompt 模板
- D. 更多消融实验
- E. 更多定性案例与失败分析
- F. 社会影响、部署注意事项与隐私问题
- G. 可复现性清单
