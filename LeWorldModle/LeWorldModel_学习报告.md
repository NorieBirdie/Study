# LeWorldModel 阶段性学习报告

## 一、论文理解

本次学习的论文是 **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**，论文关注的是从原始像素中学习稳定的 latent world model。它不是让模型直接在像素空间生成未来图像，也不是依赖大规模预训练视觉编码器，而是希望通过 JEPA，也就是 Joint-Embedding Predictive Architecture，在低维 latent 空间里学习环境动力学。

这篇论文要解决的核心问题，不是单纯证明世界模型能够预测未来，而是探索一种简单、稳定、端到端的训练方式，让模型可以直接从像素中学到不会 collapse 的世界模型。很多 JEPA 方法只需要预测未来表示，不需要重建所有像素细节，形式上比较简洁；但真正训练时很容易出现 representation collapse，也就是编码器把不同输入都映射到相同或相近的 latent，预测损失虽然变小了，但表示已经失去意义。

过去解决 collapse 的方式通常比较复杂。例如 V-JEPA / I-JEPA 一类方法会使用 stop-gradient、EMA target encoder 等训练技巧；DINO-WM 这类方法会直接使用冻结的 DINOv2 预训练特征，绕开端到端训练的不稳定；PLDM 则使用类似 VICReg 的多项正则损失，但代价是损失项很多、超参数很多、训练过程不够简洁。LeWorldModel 的目标就是把这些东西压缩成一个更干净的方案：只保留一个下一步 latent 预测损失，再加一个防止 latent collapse 的 SIGReg 正则。

从这个角度看，LeWorldModel 的贡献可以概括为三点。第一，它实现了从 raw pixels 到 latent dynamics 的端到端 JEPA 训练，不需要冻结预训练编码器，也不需要 reward、state 或 reconstruction loss。第二，它把防 collapse 的机制简化为让 latent embedding 服从各向同性高斯分布，训练目标只有两项，真正需要调的超参数主要是正则权重 `lambda`。第三，它虽然模型规模很小，大约 15M 参数，但在多个 2D / 3D 连续控制任务上仍然能保持有竞争力的规划性能，并且规划速度明显快于依赖 foundation model 特征的世界模型。

这篇论文比较重要的一点是，它没有把世界模型做成大规模视频生成器，而是围绕控制和规划需求来设计 latent world model。对于控制任务来说，模型不一定需要生成高保真的未来图像，更关键的是 latent 能够预测未来、支持目标匹配和动作规划，并且在 latent 空间中保留足够的物理结构。这样一来，世界模型的重点从视觉生成转回到控制所需的动态预测。

---

## 二、方法机制

LeWorldModel 的训练数据是离线轨迹，形式上包括原始图像观测 `o_1:T` 和动作 `a_1:T`。这里没有 reward，也没有任务标签，数据不要求是最优轨迹，只要覆盖了足够的环境动力学即可。因此它的学习目标不是直接学一个特定任务的策略，而是先学一个可用于后续规划的通用 latent dynamics。

模型主要由两个部分组成：

| 模块 | 作用 |
|---|---|
| Encoder | 把当前像素观测 `o_t` 编码为低维 latent 表示 `z_t` |
| Predictor | 根据当前 latent `z_t` 和动作 `a_t` 预测下一时刻 latent `z_{t+1}` |

论文中 encoder 使用 ViT-Tiny，大约 5M 参数；predictor 是一个 transformer，大约 10M 参数。encoder 取最后一层 `[CLS]` token，再经过带 BatchNorm 的一层 MLP 投影到 latent 表示空间。这个投影层比较关键，因为 ViT 最后一层 LayerNorm 会影响 SIGReg 正则的优化，所以需要额外投影。动作通过 AdaLN 注入 predictor，每层的 AdaLN 参数初始化为 0，让动作条件逐渐参与训练，避免一开始训练不稳定。

训练目标非常直接：

```text
L_LeWM = L_pred + lambda * SIGReg(Z)
```

其中 `L_pred` 是下一步 embedding 预测误差：

```text
L_pred = || pred(z_t, a_t) - z_{t+1} ||^2
```

如果只有这一项，模型很容易 collapse，因为最简单的解就是 encoder 对所有图像都输出同一个 latent，这样 predictor 也很容易预测。LeWorldModel 的关键在于第二项 SIGReg。它要求 batch 中的 latent embedding 分布接近各向同性高斯分布 `N(0, I)`，从而强制不同样本在 latent 空间中保持足够的多样性。

SIGReg 的具体做法不是直接在高维空间里做高斯分布匹配，而是把 embedding 投影到很多随机一维方向上，在每个一维投影上做 Epps-Pulley normality test。根据 Cramer-Wold theorem，如果所有一维投影都匹配目标分布，那么整体高维分布也会匹配。这样做的好处是计算更稳定，也更容易扩展到高维 latent。

SIGReg 的作用不是简单地让每个维度有方差，也不是只做 decorrelation，而是直接约束整个 latent 分布接近标准高斯。这样可以避免编码器把所有样本压到一个点，也避免表示空间出现明显退化。论文里强调，SIGReg 的随机投影数量 `M` 对结果影响不大，默认使用 `M=1024`，真正需要调的主要是正则权重 `lambda`，默认是 `0.1`。

---

## 三、规划链路理解

训练好世界模型后，LeWorldModel 不是直接输出动作，而是在 latent 空间里做规划。给定初始观测 `o_1` 和目标观测 `o_g`，模型先把它们编码为 latent：

```text
z_1 = enc(o_1)
z_g = enc(o_g)
```

然后初始化一段候选动作序列 `a_1:H`，用 predictor 在 latent 空间中滚动预测未来：

```text
z_hat_{t+1} = pred(z_hat_t, a_t)
```

规划目标是让预测 rollout 的最后 latent 尽量接近目标 latent：

```text
C = || z_hat_H - z_g ||^2
```

这个过程本质上是一个有限时域最优控制问题。论文使用 Cross-Entropy Method，也就是 CEM，来优化动作序列。CEM 会反复采样一批候选动作序列，用世界模型预测每条动作序列的未来 latent，选出 cost 最低的一部分 elite plans，再用这些 elite 更新采样分布。经过多轮迭代后，得到一段较好的动作计划。

这里有一个很重要的工程含义：LeWorldModel 不需要在测试时生成完整视频，只需要在 latent 空间里 rollout。相比 DINO-WM 这类使用大规模预训练视觉特征的方法，LeWM 的 token 数和模型规模都小很多，因此规划速度快得多。论文中报告的规划时间大约是 LeWM `0.98s`，DINO-WM `47s`，也就是接近 `48x` 的速度差距。

为了减少长 horizon 自回归预测带来的误差累积，论文采用 MPC 思路。也就是先规划一段动作，执行一段后重新观察环境，再基于新的观测重新规划。这个思路和很多具身智能系统里的闭环控制很类似：世界模型负责短期预演，而不是一次性把完整长任务想完。

从复现角度看，这条链路里最关键的检查点有四个：

| 检查点 | 需要确认的问题 |
|---|---|
| latent 是否稳定 | embedding 是否 collapse，SIGReg loss 是否正常下降 |
| predictor 是否学到动力学 | 给定动作后，未来 latent 是否能跟真实未来 latent 接近 |
| latent goal 是否可规划 | ||z_hat_H - z_g||` 是否真的对应任务接近程度 |
| CEM / MPC 是否有效 | 优化出的动作序列是否能在环境中闭环执行，而不是只在 latent 空间里好看 |

这也说明，LeWorldModel 的难点不只在训练 loss，还在 latent cost 是否和真实控制目标一致。如果 latent 空间保留了位置、姿态、物体关系等物理信息，那么目标匹配就比较可靠；如果 latent 忽略了任务关键变量，CEM 优化再充分也可能得到错误动作。

---

## 四、实验结果与分析

论文在四类环境上评估 LeWorldModel，包括 Two-Room、Reacher、Push-T 和 OGBench-Cube。它们分别覆盖简单 2D 导航、2D 机械臂到达、2D 推块操作和 3D 机械臂操作。所有环境都是连续动作空间，输入来自像素，目标是从初始观测规划到未来目标观测。

总体结果显示，LeWM 在 Push-T 和 Reacher 上明显强于 PLDM 和普通 DINO-WM，在 OGBench-Cube 上低于 DINO-WM，但仍然有竞争力。在 Two-Room 这种非常简单、低维度的数据环境中，LeWM 反而不占优。论文给出的解释是，SIGReg 会推动 latent 匹配高维各向同性高斯，但 Two-Room 的真实状态变化维度很低、数据多样性也低，这时强行匹配高维高斯可能会让 latent 结构不够自然。

比较典型的结果如下：

| 环境 | LeWM 表现 | 主要结论 |
|---|---|---|
| Push-T | 成功率高于 PLDM 和 DINO-WM | 像素端到端 latent dynamics 能捕捉推块中的关键物理变量 |
| Reacher | 高于 PLDM 和 DINO-WM | 对连续控制和短期动力学预测有效 |
| OGBench-Cube | 略低于 DINO-WM | 3D 视觉复杂度更高，预训练视觉特征仍有优势 |
| Two-Room | 低于部分 baseline | SIGReg 在低多样性、低内在维数据上可能不占优 |

这组结果说明，LeWM 并不是在所有环境中无条件更好。它的优势来自端到端训练、小模型、训练稳定和快速规划；但当环境视觉复杂度很高时，大规模预训练编码器仍然可能提供更强的先验；当环境过于简单时，高斯正则和真实状态分布之间也可能存在张力。因此 LeWM 更适合被理解成一种训练稳定、成本较低的世界模型路线，而不是直接替代所有 foundation-model-based 方法。

论文还做了固定计算量下的比较。在相同 FLOPs 预算下，LeWM 在 Push-T 和 OGBench-Cube 上都明显优于 DINO-WM。这一点很关键，因为实际机器人系统往往受实时性限制。如果一个世界模型每次规划都要几十秒，即使最终成功率高，也很难直接用于闭环控制。LeWM 的小模型和低维 latent 让它更接近实时控制需求。

---

## 五、物理理解与 latent 表示分析

除了控制成功率，论文还重点分析了 LeWorldModel 的 latent 空间是否真的包含物理结构。这一点比单纯看 success rate 更能说明世界模型的质量，因为世界模型的核心价值不只是完成任务，还在于它是否学到了环境中可预测、可控制的状态变量。

第一类分析是 probing。论文训练线性 probe 和 MLP probe，从 latent embedding 中预测物理量，比如 Push-T 中 agent 位置、block 位置、block 角度，OGBench-Cube 中机械臂末端位置、方块位置、关节状态等。结果显示，LeWM 在 Push-T 上整体优于 PLDM，并且和 DINO-WM 接近；在 OGBench-Cube 上，LeWM 对位置类变量表现很好，但对旋转、yaw、速度等细粒度动态变量不如 DINO-WM。

这说明 LeWM 的 latent 空间确实保留了很多任务相关物理信息，尤其是位置和空间关系。但它也不是万能的，紧凑 latent 对细粒度姿态、旋转和高频动态信息的保留仍然有限。这个现象和论文里的可视化也一致：在 OGBench-Cube 的 imagined rollout 中，整体场景结构和方块位移能保持，但机械臂末端角度等细节会逐渐丢失。

第二类分析是 decoder visualization。论文没有用 reconstruction loss 训练 LeWM，但额外训练了一个轻量 decoder，把 latent 解码回图像。结果显示，随着训练推进，decoder 能从 192 维 `[CLS]` latent 中恢复出越来越清晰的场景结构。这说明即使训练目标没有要求重建像素，latent 仍然保留了足够多的环境状态信息。

第三类分析是 violation-of-expectation，也就是意外度测试。论文设计了两种扰动：一种是视觉扰动，例如物体颜色突然改变；另一种是物理扰动，例如物体或 agent 突然 teleport 到另一个位置。结果显示，LeWM 对物理不连续事件会产生明显的 surprise spike，而对颜色变化的反应弱得多。这说明模型更敏感的是动力学和物理连续性，而不是表面视觉变化。

这一点说明 LeWM 学到的并不是单纯的图像相似性。如果模型只关注像素外观，颜色突变应该也会引起很强的 surprise；但论文结果中，teleport 这类破坏物理连续性的事件更容易被检测出来，说明 latent dynamics 对状态如何随动作合理演化有一定建模能力。

---

## 六、和其他世界模型路线的比较

把 LeWorldModel 放到更大的世界模型背景下看，它处在生成式世界模型、预训练特征世界模型和端到端 JEPA 之间。

生成式世界模型，例如 Dreamer、IRIS、DIAMOND 等，通常会在像素或离散 token 空间里生成未来观测。这类方法的优点是直观，可以作为 learned simulator 使用；但缺点是生成视觉细节成本高，而且很多像素细节未必对控制有用。LeWM 则完全避开像素重建，直接在 latent 空间预测未来，因此效率更高。

DINO-WM 这类方法使用冻结的 DINOv2 特征作为视觉表示。它的优点是稳定，因为预训练特征本身不会 collapse，而且大规模视觉预训练带来很强的先验；缺点是表示能力被预训练 encoder 限制，不能完全针对具体环境动力学端到端调整，规划成本也更高。LeWM 的优势是从像素端到端学习，模型小，规划快；劣势是在复杂 3D 视觉中，缺少大规模预训练先验可能会吃亏。

PLDM 和 LeWM 最接近，都是端到端 JEPA 路线。但 PLDM 使用多项 VICReg 风格正则，论文中提到它有七项训练目标，超参数搜索复杂度高，训练曲线也更噪。LeWM 则把目标压缩成 `prediction + SIGReg` 两项，训练曲线更平滑，超参数更少。这是 LeWM 相比 PLDM 的主要优势。

简单对比如下：

| 路线 | 代表方法 | 优点 | 问题 |
|---|---|---|---|
| 生成式世界模型 | Dreamer / IRIS / DIAMOND | 能生成未来观测，可作为 learned simulator | 像素生成成本高，视觉细节不一定服务控制 |
| 预训练特征世界模型 | DINO-WM | 表示稳定，视觉先验强 | 非端到端，规划慢，受预训练特征限制 |
| 多项正则端到端 JEPA | PLDM | 不依赖预训练 encoder | 损失复杂，超参数多，训练不够稳定 |
| SIGReg 端到端 JEPA | LeWM | 简洁、稳定、小模型、规划快 | 对数据多样性和复杂视觉仍有依赖，长 horizon 有误差累积 |

LeWM 的定位不是追求最大规模的世界模型，而是构建足够简单、可训练、可规划的 latent dynamics 模型。这个定位很有价值，因为很多具身智能系统真正需要的不是一个能生成高质量视频的大模型，而是一个能快速预演动作后果、支持闭环规划的模型。

---

## 七、局限性与后续思考

论文自己也指出了几个限制。第一，LeWM 目前更适合短 horizon 规划。因为 predictor 是自回归 rollout，预测误差会随着 horizon 增长不断累积。虽然 MPC 可以缓解这个问题，但如果任务需要很长时间尺度的推理，仍然需要层级世界模型或更强的长期规划机制。

第二，LeWM 依赖离线数据覆盖。如果训练数据没有覆盖足够丰富的状态转移，模型就很难学到可靠的动力学。尤其是在真实机器人场景中，数据分布通常会有偏差，某些动作或状态很少出现，这会直接影响 latent planning 的可靠性。

第三，SIGReg 在低多样性环境中可能不总是最优。Two-Room 的结果说明，当环境真实状态维度很低时，强制 embedding 匹配高维各向同性高斯并不一定带来更好的规划结构。这提示后续可以考虑自适应 latent 维度，或者让正则目标更贴合环境内在维度。

第四，LeWM 仍然需要动作标签。论文提到未来可以用 inverse dynamics 来减少对动作标注的依赖。这个方向对真实视频数据尤其重要，因为互联网上有大量无动作标签视频，如果能从视频中自动推断潜在动作或可控因素，世界模型的训练数据规模会大很多。

结合具身智能任务来看，LeWorldModel 可以作为一个中间模块使用：高层 VLM 负责理解语言和选择子目标，LeWM 负责在 latent 空间里评估候选动作或短期轨迹的后果，低层控制器负责把动作稳定执行到机器人或仿真器中。这样它不需要独立完成完整任务，而是作为短期预演器嵌入到更大的导航或操作系统里。

---

## 八、阶段性总结

通过这篇论文，JEPA 世界模型的关键问题更加清晰。JEPA 可以被理解成一种预测未来表示的自监督框架，但 LeWorldModel 强调的重点是：预测未来表示本身并不难，难的是如何避免 collapse，并且让学到的 latent 真的能支持控制和规划。

LeWorldModel 的核心思路可以概括成一句话：用一个端到端训练的 encoder 把像素压到紧凑 latent，用 action-conditioned predictor 预测下一步 latent，再用 SIGReg 保证 latent 分布不 collapse，最后在 latent 空间中用 CEM / MPC 做目标条件规划。这个流程相比生成未来图像更轻，相比冻结 DINOv2 特征更端到端，相比 PLDM 更稳定简洁。

这篇论文最值得借鉴的是它对简单性的强调。它没有堆很多损失项，也没有依赖复杂训练技巧，而是把世界模型训练问题收敛到两个目标：未来要可预测，表示要不坍塌。对于后续做 embodied AI 或导航/操作系统来说，这种小模型 latent world model 可能更容易嵌入实际闭环系统，尤其适合作为候选动作评估器、短期轨迹预演器或低成本 MPC 模块。

同时也需要注意，LeWM 目前还不是长程推理的完整答案。它在复杂视觉、长 horizon、数据覆盖不足和细粒度姿态建模上仍然有局限。更合理的方向可能是把它和高层语义规划、层级控制、预训练视频表示结合起来：高层负责目标和任务阶段，LeWM 负责短期物理后果预测，低层控制器负责稳定执行。这样它的价值会更加明确，也更接近真实具身智能系统中的世界模型角色。
