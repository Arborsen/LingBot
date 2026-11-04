# RTC：动作块的异步推理与连续执行

本项目推理侧采用的方案参考Physical Intelligence于 **2025年6月9日** 发布的[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)，作者为Kevin Black、Manuel Y. Galliker、Sergey Levine。[项目页面](https://www.pi.website/research/real_time_chunking)提供原理与演示。

本文记录本项目采用的推理方法和LingBot接口映射。本次开源快照未包含RTC专用运行代码和独立性能对照；[公开成功率](../experiments/benchmarks/step22000_step25000/README.md)来自固定50步同步执行基线。RTC方法归属原论文，下面的时序与公式是应用说明。

## 三种长度

| 配置 | 基线值 | 单位与作用 |
| --- | ---: | --- |
| `chunk_size` / `n_action_steps` | 50 | 预测时域：一次生成50个动作目标 |
| `use_length` | 50 | 执行时域：基线把预测块的前50个动作交给机器人 |
| `num_steps` | 10 | 生成时迭代次数：每次更新整个动作块，不是机器人执行10步 |

配置定义见[configuration_lingbot_vla.py](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py)，执行截取见[lingbot_vla_v2_policy.py](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py)。具体部署入口可能覆盖默认值，因此公开结果另外记录实际运行配置。

## 为什么需要跨块引导

同步动作分块会在执行完旧块后等待新推理，推理延迟在真实控制系统中可能表现为停顿。仅把推理移到后台，又可能在切换时跳到另一条运动策略，出现位置、速度或夹爪切换的不连续。

RTC同时处理两个问题：控制端继续执行旧块，推理端提前生成新块；新块的生成受旧块已承诺执行动作的约束，并在后续重叠区逐渐释放约束。这是推理时的引导，模型参数保持固定。

## 时间对齐：第20步推理，第30步接续

下面使用从0开始的下标，假设旧块覆盖全局控制时刻`0…49`，一次推理占用10个控制周期：

```text
全局时刻       0 ........ 19 | 20 ........ 29 | 30 ........ 49 | 50 ...
控制端         执行旧块      | 继续执行旧块    | 执行新块        | ...
推理端                     观测o20并开始生成 ------> 新块可用
新块索引                    0 ........ 9  | 10 ........ 29 | 30 ...
约束来源                    旧块20…29     | 旧块30…49作软约束
```

新块是以观测时刻20为起点，覆盖全局时刻`20…69`。它的前10个动作对应推理期间已经执行的旧动作；在时刻30接续时，应从**新块索引10**开始，不能重新执行新块索引0。

因此，预测50步、执行到20步开始重规划、在30步换块可以同时成立。第20步和第30步是示例调度点；实际接续索引必须根据观测时间戳、已执行游标和测得的推理延迟计算。

## Flow Matching外的引导

先采用论文中从噪声到动作的记号：生成时间为$\tau\in[0,1]$，速度网络为

$$v_\theta(A^\tau,o,\tau).$$

将旧块未执行部分按照全局控制时间对齐到新块，得到目标$Y$。对当前生成状态作一次终点估计：

$$\hat A=A^\tau+(1-\tau)v_\theta(A^\tau,o,\tau).$$

用一个简化的加权误差说明引导方向：

$$E=\frac12\sum_i w_i\lVert\hat A_i-Y_i\rVert^2.$$

推理期间必然执行的前缀使用最高约束权重；后续重叠区域逐渐降低权重；更远的未来权重为0，允许重新规划。对应的方向修正可写为

$$v_{\rm RTC}=v_\theta-\lambda(\tau)\nabla_{A^\tau}E.$$

这是解释方向的简化写法；论文的具体引导系数、软掩码和权重截断应按原算法实现。求梯度的对象是**当前动作张量**，不是更新模型参数。误差通过终点估计反传，包含速度网络对动作输入的依赖；仅在最后对两块动作求平均不能替代这个生成过程。

## 与LingBot时间方向的映射

当前源码的[FlowMatchingV2.sample_actions](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py)从`t=1`的噪声积分到`t=0`的动作，采用`dt=-1/num_steps`，与上面的$\tau$方向相反。

对应关系为$t=1-\tau$，在本地记号下：

$$\hat x_0=x_t-t\,v_\theta(x_t,o,t).$$

若沿用上述误差定义，示意性的修正速度为$v_\theta+\lambda(t)\nabla_{x_t}E$，再用负的`dt`更新。不能原样复制相反时间方向的符号。

接入时需要处理：

1. **坐标与归一化。** 旧动作目标$Y$和正在生成的动作必须处于相同的归一化空间、通道顺序和时间坐标。机器人有效动作维度与模型padding维度分开掩码。
2. **缓存与梯度范围。** 观测前缀和KV cache可按原路径构建；引导阶段保留动作输入的梯度通路，固定模型权重。外层`inference_mode`、编译路径和缓存是否允许该VJP，需要单独检查。
3. **逐次生成的引导。** 在10次Flow Matching迭代中更新约束；旧动作由采样器接收，原速度网络的输入结构可以保留。
4. **异步执行。** 推理工作线程读入不可变的观测、旧块和游标快照；控制端持续消费旧动作；完成后按实际已执行长度丢弃新块中过期的前缀。
5. **晚到结果。** 旧块快耗尽、推理超时或新块已过期时，执行端必须有明确的调度处理，不能重放已执行动作。

概念接口可写成以下伪代码，并非当前公开部署入口的可执行API：

```python
new_chunk = rtc_sample(
    observation=latest_observation,
    aligned_old_actions=remaining_old_actions,
    overlap_weights=weights,
    inference_delay_steps=estimated_delay,
    prediction_horizon=50,
    flow_iterations=10,
)
handoff_offset = executed_cursor_now - observation_cursor
action_queue.replace_future(new_chunk[handoff_offset:])
```

## 如何验证RTC收益

同一checkpoint、相同场景/指令与推理seed，分别比较固定50步同步基线、无引导的异步块切换和RTC。记录成功率、墙钟耗时、控制周期、推理延迟分位数、动作队列耗尽次数，以及块边界的位置/速度变化和夹爪切换。

RoboTwin的同步评测可能在等待推理时暂停物理步进。测实时收益时，应明确仿真时间与墙钟时间的关系、推理延迟期间是否继续控制；不能仅凭仿真墙钟变快就宣称真实控制动态或成功率已经改善。

RTC专项对照结果目前未归档，因此本项目没有发布RTC带来的成功率增益、吞吐倍数或延迟改善数字。
