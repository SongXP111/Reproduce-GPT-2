# Reproduce-GPT-2

Following [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=ggCyvU22Hmjm-EEW)

---

## 📚 AI Infra 与模型架构核心问答库

本栏目专门记录在复现 GPT-2 过程中探究的核心技术问题。这些问题涵盖了从大模型算法直觉到 GPU 底层系统优化的核心要点。

为了保持 README 的简洁和易读，我们已将所有核心问答整理至独立的深度学习知识库文档中：
👉 **[AI Infra 与模型架构核心问答库 (QA.md)](QA.md)**

### 问答库包含的核心技术问题索引：
1. **GELU 是什么？** 为什么它可以让 Token 加深对自己的理解？
2. **单文件组织形式**：我们有这么多 Class，为什么不把每个 Class 放在一个单独的文件里？
3. **整除断言**：`assert config.n_embd % config.n_head == 0` 这个断言是什么意思？
4. **view 与转置**：view 是做什么的？为什么在转置之后必须加上 `.contiguous()` 才能使用 view？
5. **KV Cache 机制**：在自注意力机制中，Q, K, V 都是整个 Token 串的吗？（深入推理优化）
6. **算子合并**：c_attn 是什么？为什么不拆成三个独立的 q_proj、k_proj、v_proj？
7. **多头重塑与转置**：多头注意力的重塑（view）与转置（transpose）是为了什么？
8. **自注意力机制原理**：`scaled_dot_product_attention` 的算法数学、因果屏蔽与 Flash Attention 硬件优化原理。
9. **正交向量空间设计**：为什么词向量（wte）和位置向量（wpe）可以直接相加？难道不会把信息混成一团吗？
10. **wte sharing (权重共享)**：什么是 wte sharing（权重绑定/共享）？它有什么作用？
11. **LayerNorm 梯度稀释陷阱**：既然模型中经过了 LayerNorm，为什么在深层网络中还会发生方差爆炸和训练崩溃？
12. **nn.Embedding 的物理本质**：`nn.Embedding` 在物理上到底是什么操作？它和 Linear 有什么区别？
13. **残差连接（Residual Connection）**：残差连接到底解决了什么问题？梯度高速公路原理。
14. **LayerNorm 的数学本质**：LayerNorm 在数学上到底做了什么？LayerNorm vs BatchNorm。
15. **Tokenization 与 BPE**：tiktoken 是怎么把文字变成数字的？字符级 vs 词级 vs BPE 子词。
16. **Tensor 与 .to(device)**：张量是什么？设备转移在物理上发生了什么？CPU/CUDA/MPS 对比。
17. **Cross Entropy Loss**：Cross Entropy Loss 到底在计算什么？为什么初始 Loss ≈ log(vocab_size)？
18. **Logits 与 Softmax**：Logits 是什么？它和概率有什么区别？为什么不直接输出概率？
19. **梯度下降与反向传播**：模型是怎么"学习"的？训练循环四步骤的物理含义。
20. **AdamW 优化器**：为什么选 AdamW 而不是 SGD？动量、自适应学习率、解耦权重衰减。
21. **_init_weights 完整原理**：0.02 这个魔术数字的来龙去脉，NANOGPT_SCALE_INIT 残差缩放机制。
22. **Kernel Fusion（算子融合）**：算子融合是什么？为什么它能大幅加速 GPU 计算？
23. **FlashAttention 加速机制**：FlashAttention 是什么？我需要理解到什么程度？
24. **词表对齐优化**：为什么我们要把词表大小 vocab_size 从 50257 改为 50304？
25. **AdamW 优化器超参**：AdamW 优化器中 betas=(0.9, 0.95) 和 eps=1e-8 的物理含义与作用是什么？
26. **梯度裁剪（Gradient Clipping）**：clip_grad_norm_ 的底层原理与作用是什么？如何防止训练崩溃。
27. **线性预热与余弦衰减**：学习率调度器中的线性预热（Linear Warmup）与余弦衰减（Cosine Decay）的物理原理与作用是什么？
28. **优化器分组与融合**：configure_optimizers 函数的作用是什么？为什么要做参数衰减分组与融合优化器优化？
29. **梯度累积与分布式通信**：梯度累积（Gradient Accumulation）在 AI Infra 视角的底层设计与要点是什么？
30. **梯度累积与 Loss 归一化**：梯度累积时，为什么要在 backward() 之前执行 loss = loss / grad_accum_steps？
31. **DDP 分布式数据并行初始化**：DDP 初始化代码的底层原理是什么？RANK、LOCAL_RANK、WORLD_SIZE 分别代表什么？
