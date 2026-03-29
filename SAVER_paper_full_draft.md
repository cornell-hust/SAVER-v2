# SAVER: Search, Alert, and Verify for Early Video Anomaly Understanding

## Abstract

Long-video anomaly understanding is not only a recognition problem but also an operational decision problem: under limited observation budget, a system must decide where to look, when an anomaly becomes actionable, and whether the currently selected evidence is sufficient to justify an alert. Existing video anomaly detection and anomaly-understanding pipelines remain largely passive, reasoning over densely observed clips or improving outputs only after relevant content has already been seen. We present **SAVER**, a tool-using active agent for early video anomaly understanding. SAVER is externally query-free at the task interface, but it is allowed to generate internal hypothesis-conditioned search queries during rollout to inspect candidate evidence more effectively. The policy acts through five tools, `scan_timeline`, `seek_evidence`, `emit_alert`, `verify_hypothesis`, and `finalize_case`, and is coupled with a multi-view counterfactual evidence verifier that evaluates `full`, `keep`, `drop`, and `alert_prefix` views to test evidence sufficiency, evidence necessity, and alertability at the issued time. We train SAVER in two stages. First, we perform schema-driven multi-turn oracle supervised fine-tuning with explicit verifier-feedback branches. Second, we apply **Counterfactual Evidence-and-Alert GRPO (CEA-GRPO)**, which augments grouped rollout advantages with search-local, alert-local, and evidence-local counterfactual credit before token-level clipped policy optimization against a frozen old policy and KL regularization toward a fixed reference policy. We also define a temporal-first SAVER schema centered on anomaly intervals, precursor intervals, earliest-alert supervision, evidence moments, and counterfactual annotations. The current release and experiments focus on an MSAD-based instantiation, while the schema is designed to extend to additional anomaly-understanding sources. Together, these choices recast anomaly understanding from passive recognition over observed clips into active search, explicit alerting, and verifier-aware decision making under budget.

## 1. Introduction

Classical video anomaly detection (VAD) asks a relatively narrow question: is a frame, clip, or video abnormal? This formulation has driven substantial progress through reconstruction, future prediction, weak supervision, and more recently open-vocabulary recognition [1-7]. In deployment, however, a useful surveillance system must do more than assign abnormality scores. It should determine what happened, localize when it unfolded, warn as early as possible, and explain the warning with evidence that a human operator can inspect. This broader requirement motivates the transition from anomaly detection to **video anomaly understanding** (VAU) [7-13].

What makes this transition difficult is not only richer semantics, but also the sequential nature of long surveillance video. Anomalous content is sparse relative to duration, so dense high-cost observation is inefficient. More importantly, early warning is an online decision problem: before the event has fully unfolded, the system must decide whether the evidence seen so far already justifies intervention. Explanation is also not merely a language-generation problem. A rationale is only trustworthy if it is grounded in evidence whose removal would materially alter the anomaly prediction, temporal grounding, or alert decision. These constraints suggest that anomaly understanding should be formulated not only around *what* to predict, but around *how to search*, *when to alert*, and *whether the currently selected evidence is actually sufficient*.

This perspective separates our target problem from two nearby but different lines of work. First, passive LVLM-based anomaly reasoning methods improve semantic outputs over already observed video, but they do not treat observation itself as a learned decision process. Second, recent RL-based anomaly reasoning methods improve reasoning behavior over long videos, yet they still often leave the interaction protocol underspecified. In our setting, there is no anomaly-localizing external query at the task interface. The anomaly may occur anywhere, or may not exist at all. The agent must autonomously discover suspicious evidence, decide whether that evidence already supports an alert, and keep searching when the current claim is not yet sufficiently verified.

We therefore take the following position: **early video anomaly understanding is an active decision problem, not a passive post-hoc reasoning problem**. SAVER, short for **Search, Alert, and Verify**, is built around this claim. Instead of a proposal-centric architecture, the current implementation adopts a tool-centric active-agent formulation. The agent can first inspect global context with `scan_timeline`, then issue targeted `seek_evidence` calls driven by internally generated hypotheses, explicitly record tentative or final alarms through `emit_alert`, validate the current claim with `verify_hypothesis`, and only then commit a structured decision via `finalize_case`. This formulation makes alert timing, evidence sufficiency, and anomaly understanding part of the same sequential policy loop.

On top of this agent loop, SAVER introduces a verifier that is better aligned with the operational question we actually care about. The verifier does not only ask whether the full searched trajectory supports the claim. It contrasts `full`, `keep`, and `drop` evidence views to test sufficiency and necessity, and it additionally scores an `alert_prefix` view to decide whether the alert was already actionable at the issued time. This produces two coupled verdicts: a **primary evidence status** and an **alert status**. In this way, the verifier evaluates both *claim support* and *alertability*.

The training narrative must also follow the implemented system rather than an aspirational one. SAVER is first warm-started from schema-driven oracle tool trajectories produced from annotated evidence moments, and is then improved with CEA-GRPO, which combines grouped rollout scoring with search-local, alert-local, and evidence-local counterfactual credit before token-level clipped optimization and fixed-reference KL regularization. This design keeps the policy inside a stable tool-using action manifold while still letting reward improve evidence search, alert timing, and verifier-aware behavior.

Finally, the benchmark story should stay grounded in the current release. SAVER uses a temporal-first annotation schema with anomaly intervals, precursor intervals, earliest-alert supervision, evidence moments, and counterfactual annotations. The schema is intended to support multiple anomaly-understanding sources, but the current released agent conversion and experimental pipeline are centered on MSAD. The central thesis of this paper is therefore simple: **SAVER turns anomaly understanding from passive recognition over observed clips into externally query-free, verifier-aware evidence acquisition under limited budget**. Concretely, we make three contributions. First, we formalize early anomaly understanding as an active tool-using decision problem in which the task itself is externally query-free, while internally generated search hypotheses are allowed during rollout. Second, we propose SAVER, a unified `scan -> search -> alert -> verify -> finalize` framework centered on a multi-view counterfactual evidence verifier that evaluates sufficiency, necessity, and alertability together. Third, we define a temporal-first SAVER annotation schema and instantiate the current release on MSAD with a dual-track evaluation protocol that keeps primary end-task metrics reference-free while reserving reference-conditioned verifier outputs for training and diagnostic analysis.

## 2. Related Work

### 2.1 Classical and Weakly Supervised Video Anomaly Detection

Early VAD methods largely modeled anomalies as deviations from learned regularity, using reconstruction or future prediction to identify events that depart from normal video dynamics [1, 2]. Weakly supervised methods later improved temporal localization in long surveillance video through multiple-instance learning, ranking objectives, and noise-robust training [3-6]. These lines of work substantially advanced anomaly discrimination and coarse localization, but they share a passive operating assumption: the model scores a predefined stream of clips rather than deciding which evidence to inspect next, when the system should warn, or whether the selected evidence is sufficient to justify the warning.

### 2.2 Open-Vocabulary and LVLM-Based Anomaly Understanding

Open-vocabulary and large-model approaches broaden anomaly analysis beyond fixed taxonomies and scalar abnormality scores. Open-Vocabulary Video Anomaly Detection [7] extends anomaly recognition to semantically richer categories. Harnessing Large Language Models for Training-free Video Anomaly Detection [18] and Follow the Rules [9] further show that language-model supervision can improve anomaly reasoning and free-form explanation over surveillance video. Towards Surveillance Video-and-Language Understanding [19] expands this direction at dataset scale by framing surveillance understanding as a richer video-language problem. Anomize [12] continues the open-vocabulary line with stronger semantic alignment. These methods are important because they move the field toward richer semantics, but they still mostly assume that the relevant content has already been observed or densely sampled. SAVER instead treats **observation itself** as part of the policy.

### 2.3 Explainable and Evidence-Grounded Anomaly Reasoning

CUVA makes a crucial step toward anomaly understanding by asking what happened, why it happened, and how the anomaly unfolds, thereby shifting attention from binary detection to structured reasoning [8]. VERA verbalizes anomaly evidence through vision-language learning [10], and Holmes-VAU studies long-horizon anomaly understanding across multiple granularities [11]. These works motivate our setting directly, yet they still leave unresolved a deployment-critical issue: explanation quality should depend not only on language quality over available video, but also on **which evidence the system actively acquired under budget**, whether that evidence remains supportive under counterfactual tests, and whether the evidence available **before the alert time** was already actionable. SAVER addresses this gap with a multi-view counterfactual evidence verifier that explicitly contrasts searched temporal evidence subsets under `full`, `keep`, `drop`, and `alert_prefix` views instead of treating explanation as language over fixed observed context.

### 2.4 Active Search and Reinforcement Learning for Long-Form Video Understanding

Sequential decision making is natural for long-form video analysis. Holmes-VAU already highlights adaptive temporal search as an important ingredient for long-horizon anomaly understanding [11]. Recent RL-based reasoning work such as VAU-R1 improves anomaly reasoning behavior through reinforcement fine-tuning [13]. PPO-style and group-relative objectives remain practical foundations for such sequential policies [13, 15]. SAVER extends this direction to a more operational setting: the policy must actively acquire evidence, issue explicit alerts, verify whether the current claim is supported, and declare normality only after enough search has been performed. Reinforcement learning is therefore used here to optimize search efficiency and decision timing under budget, not only downstream reasoning quality. In the current implementation, RL uses a **grouped-rollout, token-level clipped policy objective**: rollout rewards are converted into turn- and token-level credit, the surrogate is evaluated against a frozen old policy, and a separate KL term anchors updates to a fixed SFT reference policy for stability.

## 3. Problem Setting and SAVER Schema

### 3.1 Task Definition

We consider a long surveillance video

\[
V = \{f_t\}_{t=1}^{T},
\]

which is partitioned into clip units

\[
\mathcal{C} = \{c_i\}_{i=1}^{N}.
\]

The model interacts with the video under a finite observation budget \(B\). At step \(k \leq B\), it chooses a tool call

\[
a_k = (t_k, \alpha_k), \qquad t_k \in \mathcal{T},
\]

where

\[
\mathcal{T} =
\{\texttt{scan\_timeline},\ \texttt{seek\_evidence},\ \texttt{emit\_alert},\ \texttt{verify\_hypothesis},\ \texttt{finalize\_case}\}.
\]

The tool arguments \(\alpha_k\) specify the queried time span, the internal search hypothesis if present, the alert decision, or the structured claim to be verified. Each tool returns an observation \(o_k\), so the searched trajectory is

\[
\hat{\tau} = \{(a_k, o_k)\}_{k=1}^{K}, \qquad K \leq B.
\]

The system ultimately outputs a structured prediction

\[
\hat{\mathcal{Y}} =
(\hat{y}^{exist}, \hat{y}^{cat}, \hat{s}, \hat{I}, \hat{P}, \hat{t}_{alert}, \hat{E}, \hat{c}),
\]

where \(\hat{y}^{exist}\) denotes anomaly existence, \(\hat{y}^{cat}\) the anomaly category, \(\hat{s}\) the severity, \(\hat{I}\) the anomaly interval, \(\hat{P}\) the precursor interval, \(\hat{t}_{alert}\) the earliest actionable alert, \(\hat{E}\) the evidence subset, and \(\hat{c}\) the counterfactual type. Richer textual summaries, rationales, and QA outputs are compatible with the same schema, but the current implementation centers on **verifier-informed structured finalization** rather than a standalone language-generation module.

The defining property of the setting is therefore not only structured prediction, but **externally query-free decision making under partial observation**. There is no anomaly-localizing external question at the task interface. However, the agent is allowed to generate internal hypothesis text when calling `seek_evidence`. In other words, SAVER is **task-level external-query-free, but internally hypothesis-conditioned**.

### 3.2 Why Existing Benchmarks Are Insufficient

Most classical VAD benchmarks are built for anomaly scoring or coarse temporal localization. Even more recent anomaly-understanding resources primarily evaluate semantic explanation after relevant content has already been observed [8-13, 19]. That leaves three deployment-relevant questions only weakly supervised: can a model search efficiently rather than observe densely, can it warn early rather than only localize retrospectively, and can it justify its decision with evidence that remains decisive under counterfactual testing? These questions require explicit precursor intervals, earliest-alert supervision, evidence moments, and search-aware metrics. The SAVER schema is designed around exactly this gap.


### 3.3 Schema and Current Release

Each SAVER record follows a paper-facing schema centered on the information required for active anomaly understanding. At the temporal level, records may include an anomaly interval, a precursor interval, an earliest-alert frame, and evidence moments. At the semantic level, records include anomaly category, severity, counterfactual type and text, and key objects. At the language level, the schema can additionally store summaries, rationales, and QA supervision.

The schema is deliberately source-agnostic, but the **current code release and experiments in this paper focus on MSAD** [16]. This keeps the paper aligned with the agent conversion, rollout, scoring, and training pipeline that are already implemented. Additional sources such as CUVA [8] and ECVA can be mapped into the same schema in future releases, but we do not claim a fully unified multi-source benchmark in the present version.

| Source | Videos | Anomalous | Normal | Train | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSAD | 720 | 240 | 480 | 480 | 240 |

MSAD is an appropriate current release because it contains both anomalous and normal videos. This is crucial for measuring false alert rate, hard-normal false alerts, and the cost of premature warning behavior, all of which are central to SAVER's early-alert framing.

### 3.4 Evaluation Protocol

The evaluation protocol is organized as a **main-eval track** plus an optional **diagnostic-eval track**. This separation is important because the current training pipeline can use a reference-conditioned offline verifier for reward construction and rollout diagnosis. Such signals are useful for optimization and debugging, but they should not be conflated with primary end-task evaluation. Unless otherwise stated, the paper’s main tables report only the reference-free main-eval metrics computed from the policy’s own outputs and the ground-truth annotations.

The first main-eval group measures **final decision correctness**. We report anomaly-existence AP, anomaly-category Macro-F1, anomaly-interval mIoU, precursor-interval mIoU, and counterfactual-type accuracy. Existence AP is computed over video-level anomaly scores derived from the policy’s own finalized decision state, while Macro-F1 is computed over anomalous categories only so that performance is not dominated by the normal class. Temporal mIoU and precursor mIoU compare the finalized structured intervals against the corresponding annotated intervals.

The second main-eval group measures **alert quality**, which is central to SAVER. We report alert utility, premature alert rate, late alert rate, false alert rate, and hard-normal false alert rate. Let \(t^\star\) denote the annotated earliest actionable alert time and let \(\hat{t}\) denote the first emitted anomaly alert. We define an alert utility

\[
\mathrm{AU}(V)=
\begin{cases}
-1, & \text{if } V \text{ is normal and any anomaly alert is issued},\\
0, & \text{if } V \text{ is anomalous and no alert is issued},\\
- \min\!\left(1,\frac{t^\star-\hat{t}}{\tau_{\mathrm{pre}}}\right), & \text{if } \hat{t} < t^\star - \delta,\\
1-\min\!\left(1,\frac{\hat{t}-t^\star}{\tau_{\mathrm{late}}}\right), & \text{otherwise},
\end{cases}
\]

where \(\tau_{\mathrm{pre}}\) is the precursor duration, \(\tau_{\mathrm{late}}\) is the anomaly duration, and \(\delta\) is a small tolerance window. This metric rewards timely alerts, penalizes premature intervention, and sharply punishes false alarms on normal videos.

The third main-eval group measures **evidence faithfulness**. We report evidence precision@3, evidence recall@3, and evidence F1@3. These metrics are computed from the policy’s own selected or finalized evidence windows rather than from any offline verifier-selected subset. Evidence F1@3 is obtained by matching the finalized evidence subset against the top annotated evidence windows using one-to-one temporal matching with an IoU threshold. This metric reflects whether the model not only arrives at the right answer, but also arrives there with the right evidence.

The fourth main-eval group measures **search efficiency and protocol compliance**. We report mean inspected clip ratio, mean search steps, mean latency, tool-call validity rate, and protocol-compliance rate. The inspected clip ratio is the temporal union of visited windows divided by video duration. Protocol compliance checks whether the rollout follows the intended `scan/search -> alert/verify -> finalize -> answer` structure rather than bypassing the tool loop, while still treating an already materialized finalized case as compliant when the answer is returned. These metrics matter because SAVER is not a passive classifier; it is an active decision policy operating under limited observation budget.

The diagnostic-eval track is reported separately. It includes verifier primary-status ratios over `complete`, `incomplete`, `redundant`, and `misaligned`, verifier alert-status ratios over `justified`, `premature`, `late`, and `not_applicable`, as well as reward summaries and other reference-conditioned offline-verifier statistics. In the released code these quantities are attached only when diagnostic evaluation is explicitly enabled. Auxiliary language metrics such as BLEU, CIDEr, METEOR, and ROUGE-L can still be reported when summary or rationale generation is enabled, but they are not the primary claim of the current release. The protocol is therefore designed to make the paper’s thesis measurable without conflating training-time oracle-style diagnostics with the main end-task results.

## 4. Method

### 4.1 Overview

SAVER is organized around a tool-centric decision loop:

\[
\texttt{scan\_timeline} \rightarrow \texttt{seek\_evidence} \rightarrow \texttt{emit\_alert} \rightarrow \texttt{verify\_hypothesis} \rightarrow \texttt{finalize\_case}.
\]

This decomposition is central to the method. `scan_timeline` provides low-cost global context, `seek_evidence` performs targeted inspection of candidate moments, `emit_alert` turns actionability into an explicit decision, `verify_hypothesis` tests whether the currently selected evidence supports both the claim and the alert, and `finalize_case` commits the structured prediction only after sufficient support has been gathered. In practice, this interaction contract is schema-driven rather than handwritten: the prompt is rendered from the same tool function schemas used by the runtime, and the user message can include a configurable set of initial preview frames sampled from the video cache. This keeps the tool interface consistent across SFT, rollout, and evaluation.

### 4.2 Active Agent State and Tool Actions

We formulate SAVER as a partially observable sequential decision process

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, \gamma),
\]

but the action space is a **tool-call space** rather than a proposal-index space. At step \(k\), the state contains the visited windows \(\mathcal{W}_k\), the evidence ledger \(\mathcal{L}_k\), the emitted alerts \(\mathcal{A}_k\), the cached verifier records \(\mathcal{U}_k\), and the latest structured claim \(c_k\). The agent chooses a tool call \(a_k=(t_k,\alpha_k)\), receives an observation \(o_k\), and updates this explicit state. The first two tools govern evidence acquisition. `scan_timeline(start, end, ...)` performs a broad inspection pass and writes the resulting window into the visited-window ledger. `seek_evidence(query, start, end, ...)` performs targeted search over a candidate interval and records both the visited window and the search query. The key detail is that `query` is generated internally from the agent’s current hypothesis rather than supplied externally by a user, keeping the method externally query-free while still enabling hypothesis-conditioned search.

### 4.3 Explicit Alerting and Structured Claims

SAVER treats alerting as part of the action space. Through `emit_alert`, the agent can record `soft_alert`, `hard_alert`, or `declare_normal` decisions together with the predicted existence label, category, and alert time. A `soft_alert` is non-terminal and allows the agent to continue searching, whereas a `hard_alert` indicates that the current hypothesis is strong enough to justify commitment. `declare_normal` is reserved for cases in which the observed evidence is sufficient to support the absence of an actionable anomaly. This design is important because early warning is not equivalent to retrospective localization: the agent must decide not only what happened, but also whether it is already actionable **now**. The claim that is later verified is therefore explicitly materialized in the environment state rather than left implicit in free-form text.

### 4.4 Multi-View Counterfactual Evidence Verification

Let \(\hat{E} \subseteq \mathcal{W}_K\) denote the currently selected evidence subset. SAVER constructs four evidence views:

\[
\mathcal{W}^{full} = \mathcal{W}_K, \qquad
\mathcal{W}^{keep} = \hat{E}, \qquad
\mathcal{W}^{drop} = \mathcal{W}_K \setminus \hat{E}, \qquad
\mathcal{W}^{prefix} = \{W \in \mathcal{W}_K \mid W \text{ starts before the alert time}\}.
\]

The first three views test classical counterfactual evidence properties. `full` measures support from all searched evidence. `keep` asks whether the selected subset is sufficient on its own. `drop` asks whether removing the selected subset materially weakens support, thereby testing necessity. The additional `alert_prefix` view asks a different question: if we only keep the evidence available before the alert time, was the alert already actionable?

The verifier returns two linked verdicts. The **primary status**

\[
u^{primary} \in \{\texttt{complete}, \texttt{incomplete}, \texttt{redundant}, \texttt{misaligned}\}
\]

describes whether the selected evidence supports the claim. The **alert status**

\[
u^{alert} \in \{\texttt{justified}, \texttt{premature}, \texttt{late}, \texttt{not\_applicable}\}
\]

describes whether the alert decision was timely. `complete` means the `keep` view remains strong while the `drop` view weakens materially; `redundant` means useful support remains even after dropping the selected windows; `incomplete` means the chosen subset is not yet sufficient on its own; and `misaligned` means the selected evidence does not support the current claim. Separately, `premature` and `late` distinguish two operational failure modes that a tri-view sufficiency test alone cannot capture.

The verifier backend can be heuristic, learned, or hybrid. In the current implementation, heuristic support scores can be blended with a learned Qwen self-verifier, but the paper’s claim is at the **multi-view protocol level** rather than at the level of any single scorer family. Importantly, SAVER is a temporal evidence-level counterfactual verifier, not an object-level intervention engine.

### 4.5 Verifier-Informed Structured Finalization

Verification is part of the agent loop rather than a purely diagnostic afterthought. If a `soft_alert` is followed by `incomplete` or `misaligned`, the policy is encouraged to continue searching when budget remains. If the primary claim is supported but the alert is `premature`, the agent can delay commitment and accumulate more pre-alert evidence. If the evidence is `complete` and the alert is `justified`, the policy can finalize. In the current implementation, this takes the form of **verifier-informed structured finalization** rather than a separate evidence-refinement or free-form generation module. The verifier returns `verified_window_ids` together with a `best_effort_window_ids` fallback, and `finalize_case` commits a schema-validated structured decision based on searched evidence only.

### 4.6 Two-Stage Training

Training proceeds in two stages. **Stage 1** is schema-driven oracle supervised fine-tuning. Each example uses the same function-schema prompt contract as inference, can include a small initial preview, and supervises the assistant in the exact `<think>...</think><tool_call>{...}</tool_call>` or `<think>...</think><answer>{...}</answer>` format used at rollout time. From annotated evidence moments, we synthesize heuristic trajectories that begin with `scan_timeline`, continue with targeted `seek_evidence` calls over precursor, trigger, peak-action, or confirmation windows, and then pass through explicit alert and verification decisions before `finalize_case`.

Importantly, the oracle traces do not treat `verify_hypothesis` as a terminal stamp. After verification, the following tool observation can recommend `revise_claim`, `continue_search`, or `finalize`, depending on whether the current evidence is misaligned, incomplete, or already sufficient. This branching supervision is applied to anomalous cases, and normal cases also pass through a final verification step before `declare_normal`. The resulting warm start teaches `verify_hypothesis` as a genuine control decision inside the agent loop rather than as an anomaly-only ritual. For efficiency, these examples are materialized into a lightweight prepared-SFT format that stores frame references instead of raw images, while training enforces explicit text and vision budgets through configurable limits on retained images, recent text history, and maximum sequence length.

**Stage 2** is **Counterfactual Evidence-and-Alert GRPO (CEA-GRPO)** with both a frozen old policy and a fixed reference policy. For each source video, we sample a small group of rollout trajectories, score them offline with the reward and verifier stack, and compute a group-relative rollout advantage

\[
A_g = \frac{R_g - \mu(R)}{\sigma(R) + \epsilon}.
\]

This rollout-level signal is not the final optimization target. CEA-GRPO augments it with three types of local counterfactual groups: search groups compare using a search step against skipping it, alert groups compare alerting now against deferring the decision, and evidence groups compare keeping the selected evidence subset against alternative evidence selections. These local comparisons yield search-local, alert-local, and evidence-local advantages that are injected back into the responsible turns. The total turn advantage is therefore

\[
A^{\mathrm{tot}}_{g,k}
=
w^{G}_{g,k} A_g
+ w^{S}_{g,k} A^{\mathrm{search}}_{g,k}
+ w^{A}_{g,k} A^{\mathrm{alert}}_{g,k}
+ w^{E}_{g,k} A^{\mathrm{evidence}}_{g,k},
\]

where the weights \(w^{G}_{g,k}, w^{S}_{g,k}, w^{A}_{g,k}, w^{E}_{g,k}\) depend on the turn type. Search turns receive higher search-local weight, `emit_alert` and `verify_hypothesis` turns receive higher alert-local weight, and `verify_hypothesis`, `finalize_case`, and evidence-producing `seek_evidence` turns receive higher evidence-local weight. We then redistribute \(A^{\mathrm{tot}}_{g,k}\) across the response tokens of turn \(k\) using structure-aware weights that emphasize `tool_call` and `answer` payloads, along with key JSON fields, while downweighting boilerplate or low-information spans such as `<think>`. The resulting token-level advantages are denoted by \(A_{g,k,j}\) for token \(j\) in turn \(k\).

Let \(r_{g,k,j}(\theta)\) be the response-token importance ratio between the current policy and the frozen old policy snapshot:

\[
r_{g,k,j}(\theta)
=
\frac{\pi_\theta(y_{g,k,j}\mid x_{g,k}, y_{g,k,<j})}
{\pi_{\theta_{\mathrm{old}}}(y_{g,k,j}\mid x_{g,k}, y_{g,k,<j})}.
\]

The policy objective is then a token-level clipped surrogate

\[
\mathcal{L}_{\mathrm{clip}}
=
- \frac{1}{|\Omega|}\sum_{(g,k,j)\in\Omega}
\min\!\Big(
r_{g,k,j}(\theta)\, A_{g,k,j},
\operatorname{clip}(r_{g,k,j}(\theta), 1-\epsilon_c, 1+\epsilon_c)\, A_{g,k,j}
\Big),
\]

computed only on response tokens \(\Omega\). In parallel, we regularize the current policy toward a fixed SFT reference policy \(\pi_{\mathrm{ref}}\) using a forward KL term on the same response-token mask:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{clip}}
+ \beta\, \mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right).
\]

Overall, the current SAVER implementation uses **token-level clipped policy optimization with grouped rollouts, CEA local counterfactual credit, and a fixed-reference KL anchor**. Grouped rollouts preserve comparison structure across multiple trajectories from the same video, the frozen old policy provides the clipping baseline for stable improvement, and the fixed reference policy prevents the model from drifting away from valid tool syntax, stable `finalize_case` behavior, and the SFT-initialized action manifold. This is especially important because the reward is sparse and high-level: without the old-policy clip and reference anchor, the policy can degrade search behavior even when short-term reward appears to improve. At the same time, this remains a practical PPO/GRPO-style policy update rather than a full actor-critic formulation with a learned value head and GAE. The current code can use a reference-conditioned offline verifier during RL reward construction, so those verifier outputs are treated as training signals and diagnostic analyses rather than as primary evaluation metrics.

## 5. Experiments

Where large-scale comparison runs are still being refreshed, we leave numerical cells blank rather than report provisional values. The experiment structure and comparison logic are fixed. The experiments are organized around four questions: does SAVER improve operational anomaly understanding on the current MSAD release; which parts of the `scan -> search -> alert -> verify -> finalize` decomposition matter most; how much does verifier design contribute to decision quality; and what cost-quality tradeoff does active search achieve relative to passive observation?

### 5.1 Experimental Setup

Unless otherwise stated, experiments use Qwen3-VL models as the policy and verifier backbones, following the current implementation. The default policy warm start is `Qwen3-VL-8B-Instruct`, while larger checkpoints can be used for stronger rollout generation or teacher-style analysis. Our default setting uses LoRA adaptation with rank 16, scaling 32, dropout 0.05, bf16 weights, and gradient checkpointing. Stage 1 data are first converted into a lightweight prepared-SFT JSONL so that multi-turn examples can be replayed with frame references instead of raw images; the corresponding prompts are schema-driven and use a configurable preview budget. The released default configuration uses an 8-frame initial preview, a rollout budget of 6 tool turns, and explicit text/vision limits for long trajectories, including maximum sequence length, capped retained images, and recent-history truncation. The current RL configuration uses oracle-SFT warm start, grouped rollouts with `--num-generations 4`, CEA-GRPO local search/alert/evidence groups, rollout-to-turn-to-token credit assignment, token-level clipped policy updates against a frozen old policy snapshot, hybrid verification, and KL regularization to a fixed SFT reference checkpoint. Unless explicitly marked as diagnostic, all reported numbers come from the reference-free main-eval path rather than from the reference-conditioned offline verifier. We report results on the **MSAD current release**, which is the portion of the benchmark stack already aligned with the released conversion and training code.

### 5.2 Comparison Protocol

We compare against three baseline families. The first contains passive anomaly detectors, including weakly supervised VAD and open-vocabulary VAD systems, and tests whether active search offers a better quality-cost tradeoff than dense or fixed observation. The second contains anomaly-understanding baselines, including CUVA-style structured understanding, VERA, Holmes-VAU, and closely related passive LVLM pipelines whenever adaptation is feasible. The third contains SAVER ablations that isolate the value of active search, explicit alerting, multi-view verification, schema-driven oracle warm start, CEA-GRPO local counterfactual credit, and reference-KL regularization. Because several published baselines were not designed for externally query-free active inference, we describe adaptation details carefully and evaluate all methods under the same schema and metric definitions wherever possible.

### 5.3 Main Results on MSAD

MSAD is the primary testbed for anomaly-versus-normal discrimination and alert-risk evaluation because it contains both anomalous and normal videos. The key question is therefore not only whether SAVER detects anomalies accurately, but whether it does so **earlier**, with **better-supported evidence**, and with a **better quality-cost tradeoff under the same observation budget** than passive baselines. Verifier status ratios are analyzed separately in the diagnostic track rather than folded into the main comparison table.

| Method | Existence AP | Category Macro-F1 | Temporal mIoU | Precursor mIoU | Alert Utility | Premature Alert Rate | False Alert Rate | Evidence F1@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weakly supervised VAD baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Open-vocabulary VAD baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Explainable anomaly baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER w/o RL fine-tuning | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.4 Ablation Study

The ablation study isolates the paper's central claims rather than peripheral hyperparameters. In particular, it tests whether the gains come from active search, explicit alerting, multi-view verification, schema-driven oracle warm start, and CEA-GRPO with reference KL, rather than from prompt engineering alone. One important control removes `verify_hypothesis` before finalization. Another replaces the hybrid verifier with heuristic-only scoring. A third removes the fixed reference anchor during RL. A fourth collapses CEA-GRPO back to a global rollout objective without the local search/alert/evidence counterfactual groups. Together these controls test whether SAVER's behavior depends on the active-agent protocol itself rather than on any one prompt template.

| Variant | Temporal mIoU | Alert Utility | Evidence F1@3 | Mean Steps | Mean Inspected Clip Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full SAVER | TBD | TBD | TBD | TBD | TBD |
| w/o oracle-SFT warm start | TBD | TBD | TBD | TBD | TBD |
| global rollout GRPO without CEA local groups | TBD | TBD | TBD | TBD | TBD |
| RL without fixed reference policy | TBD | TBD | TBD | TBD | TBD |
| heuristic-only verifier instead of hybrid verifier | TBD | TBD | TBD | TBD | TBD |
| w/o `verify_hypothesis` before `finalize_case` | TBD | TBD | TBD | TBD | TBD |
| passive dense observation instead of active search | TBD | TBD | TBD | TBD | TBD |

### 5.5 Search Efficiency and Qualitative Analysis

Because SAVER is motivated by budgeted observation, efficiency must be reported alongside decision quality rather than in a separate engineering appendix. The most important search-efficiency measures are mean inspected clip ratio, mean search steps, mean latency, and protocol-compliance rate, interpreted jointly with temporal grounding and alert utility.

| Method | Mean Inspected Clip Ratio | Mean Search Steps | Mean Latency | Protocol Compliance | Temporal mIoU | Alert Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| SAVER | TBD | TBD | TBD | TBD | TBD | TBD |

The qualitative analysis is organized around three case types. The first shows successful early-warning trajectories in which SAVER raises a `soft_alert`, verifies the current claim, and then either continues searching or commits depending on verifier status. The second shows evidence correction cases, where the raw selected evidence is redundant or incomplete and the verifier identifies a better-supported subset or a best-effort fallback. The third shows failure cases such as late alerts, premature alerts, wrong evidence, or misaligned counterfactual claims. These cases matter because the paper's main claim concerns actionability and evidence faithfulness rather than stylistic fluency of the final text alone.

### 5.6 Current Release Scope and Future Extensions

The current release should be understood as an **MSAD-centered active-agent benchmark and training stack**, not yet as a finished all-source anomaly-understanding suite. The schema is designed to absorb additional sources such as CUVA and ECVA, but the currently implemented conversion, rollout supervision, and end-to-end training pipeline are aligned with MSAD. We therefore treat multi-source expansion as the next benchmark release rather than as a claim already established in the current paper.

### 5.7 Limitations

The main limitations to analyze in final experiments are delayed alerts under weak or ambiguous precursors, failure to preserve decisive evidence when several correlated windows remain plausible, and brittleness in internally generated search hypotheses when the anomaly is highly unusual. The verifier also operates on temporal evidence subsets rather than on explicit object-level interventions. Finally, the current implementation is strongest at structured finalization; richer summary, rationale, and counterfactual generation should be treated as follow-on modules rather than as already solved parts of the system. These limitations do not invalidate the framework; they define the exact frontier at which active anomaly understanding remains challenging.

## 6. Conclusion

We presented SAVER, a framework for **Search, Alert, and Verify** in early video anomaly understanding. Unlike passive anomaly pipelines that reason over already observed clips, SAVER treats observation itself as a learned decision problem under limited budget. The current implementation is best understood as an externally query-free **tool-using active agent**: it scans the timeline, performs internally hypothesis-conditioned evidence search, emits explicit alerts, validates the current claim with a multi-view counterfactual verifier, and then commits a structured final decision. Its training recipe is correspondingly practical: schema-driven oracle warm start followed by CEA-GRPO with rollout-to-turn-to-token credit assignment, token-level clipped policy optimization, and fixed-reference KL regularization. Together with the temporal-first SAVER schema, the prepared-SFT training pipeline, and the current MSAD release with reference-free main evaluation plus optional diagnostic verifier analysis, this formulation offers a concrete path from passive anomaly scoring toward active, early, and evidence-grounded anomaly understanding in long surveillance video.

## References

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

## Appendix Package for Submission Preparation

This package is intentionally manuscript-facing rather than repository-facing. It fixes the content and logic of figures, tables, and supplementary sections so that the paper can be migrated into a venue template later without changing the narrative.

### A. Figure Package

#### Figure 1. Overall Framework

Purpose: show SAVER as an active anomaly-understanding system rather than a passive VAD pipeline.

Required content:
- input long-video timeline
- tool-centric loop with `scan_timeline`, `seek_evidence`, `emit_alert`, `verify_hypothesis`, and `finalize_case`
- explicit visited-window ledger and alert state
- explicit split between `soft_alert` and `hard_alert`
- multi-view verifier with `full`, `keep`, `drop`, and `alert_prefix`
- final structured output after verifier-informed finalization

Suggested caption: *Overview of SAVER. A tool-using active agent scans the timeline, performs internally hypothesis-conditioned evidence search, emits explicit alerts, and uses a multi-view counterfactual verifier to test claim support and alertability before structured finalization.*

#### Figure 2. SAVER Schema and Current MSAD Release

Purpose: explain visually why the SAVER schema is temporal-first and what is included in the current release.

Required content:
- one horizontal video timeline
- anomaly interval
- precursor interval
- earliest-alert frame
- evidence moments with semantic roles
- side panel with severity, counterfactual, key objects, summary, rationale, and QA
- note that the current release is instantiated on MSAD and is designed to extend to additional sources later

Suggested caption: *The SAVER schema is organized around temporal supervision for active anomaly understanding, including anomaly interval, precursor interval, earliest-alert frame, evidence moments, and grounded semantic annotations. The current implementation and experiments focus on the MSAD release.*

#### Figure 3. Qualitative Search Trace

Purpose: show how active search differs from passive observation.

Required content:
- four to six search steps from one video
- selected tool call and inspected window at each step
- current decision action (`continue`, `soft_alert`, `hard_alert`)
- verifier primary and alert status after verification when applicable
- final verified or best-effort evidence subset

Suggested caption: *Example search trajectory of SAVER. The model raises a soft alert, verifies the current claim, and either continues searching or commits depending on the returned evidence and alert verdicts.*

#### Figure 4. Multi-View Verifier Case Study

Purpose: demonstrate that verification changes the final output rather than merely scoring it.

Required content:
- raw selected evidence windows
- `full`, `keep`, `drop`, and `alert_prefix` view summaries
- primary status and alert status
- best-effort or verified evidence subset
- final interval and alert decision after verification

Suggested caption: *The SAVER verifier can convert a raw searched evidence subset into a supported or best-effort final subset by jointly reasoning over sufficiency, necessity, and alertability.*

### B. Table Package

#### Table 1. Main Comparison on MSAD

Core message: SAVER should be evaluated jointly on anomaly recognition, early alert, and evidence quality under the reference-free main-eval protocol.

Columns:
- existence AP
- temporal mIoU
- precursor mIoU
- alert utility
- premature alert rate
- false alert rate
- evidence F1@3

#### Table 2. Ablation Study

Core message: test the central thesis that active search, schema-driven oracle warm start, CEA-GRPO local counterfactual credit, verifier usage, and reference-anchored RL each contribute materially.

Columns:
- temporal mIoU
- alert utility
- evidence F1@3
- mean search steps
- mean inspected clip ratio

#### Table 3. Search Efficiency Analysis

Core message: show that SAVER provides a better cost-quality tradeoff than passive observation.

### C. Appendix Skeleton

Appendix sections to prepare in the venue template:
- A. Full training details and hyperparameters
- B. Dataset schema and annotation examples
- C. Prompt templates for active-agent tool use and verification
- D. Additional ablations
- E. More qualitative cases and failure analysis
- F. Societal impact, deployment caveats, and privacy considerations
- G. Reproducibility checklist
