<div align="center">

# Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective

</div>




### Introduction

While Reinforcement Learning with Verifiable Rewards (RLVR) can enhance LLM reasoning, its training process poses a critical risk: Entropy Collapse.
This phenomenon is a rapid loss of policy diversity, stemming from the exploration-exploitation imbalance and leading to a lack of generalization.
Recent entropy-intervention methods aim to prevent the entropy collapse, yet their underlying mechanisms remain unclear.
In this paper, 

- **We propose a quantitative analysis framework for entropy change.** Building on this, the effect of entropy interventions can be unified and elucidated through token-level analysis.
Our findings point out a fundamental limitation of existing methods: they attempt to control the entropy indirectly.
By only affecting related factors, such as the advantage signal and generation probability, their effectiveness is inherently limited and could potentially fail.
    
- **To precisely stabilize entropy change**, we propose an adaptive and fine-grained reweighting method, namely Stabilizing Token-level Entropy-changE via Reweighting (STEER), that keeps per-step entropy change within a moderate band.
This approach prevents over-exploitation while ensuring robust exploration.

Our extensive experiments demonstrate that STEER significantly avoids entropy collapse, stabilizes entropy dynamics, and achieves stronger downstream performance across math reasoning benchmarks.








### Getting Started


We use exactly the same environment configurations as the official verl codebase.

* **Install:** [https://verl.readthedocs.io/en/latest/start/install.html](https://verl.readthedocs.io/en/latest/start/install.html)
* **Quick Start:** [https://verl.readthedocs.io/en/latest/start/quickstart.html](https://verl.readthedocs.io/en/latest/start/quickstart.html)

Environment setup
```bash
pip install git+https://github.com/volcengine/verl.git@v0.4.1.x
```


### Datasets
We use public dataset [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) for training, and six public math benchmarks for validation. 
All datasets are provided in folder `STEER/datasets`.


### Base Model
We use [Qwen](https://huggingface.co/Qwen/collections) series model for training.
One can download the models from huggingface, for example,
```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir Qwen2.5-Math-7B --resume-download
```


### Training
The training scripts are fully inherited from the standard GRPO training.

We provide ready-to-run scripts:

```
cd STEER/run
bash run_linear.sh
bash run_exp.sh
```

One can change $\lambda_{\text{min}}$ by tuning:

```
+actor_rollout_ref.actor.policy_loss.token_weight_min
```


Experiments in extreme senario can be conducted to test entropy control:

```
bash run_linear_extreme.sh
bash run_exp_extreme.sh
```


Core Implementation are provided in file: `STEER/verl/trainer/ppo/core_algos.py` (lines ~579–808)


Note that we run all experiments using 8 H20s.
If one want to launch distributed tasks, please refer to the instruction of [verl](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main).


### Evaluation
We provide the evaluation codebase integrated in the verl infra.
Please refer to script eval.sh for evaluation scripts on our [released model](https://huggingface.co/zzzzzzzzzzhao/STEER/tree/main).
```
cd STEER/run
bash eval.sh
```



## Acknowledgement

We build on [verl](https://github.com/volcengine/verl) and qwen math-reasoning evaluation protocols.
All competitors in can be easily implemented or are already implemented in [verl](https://github.com/volcengine/verl).

---


<!--


## Citation

```bibtex
@article{hao2025rethinking,
  title={Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective},
  author={Hao, Zhezheng and Wang, Hong and Liu, Haoyang and Luo, Jian and Yu, Jiarui and Dong, Hande and Lin, Qiang and Wang, Can and Chen, Jiawei},
  journal={arXiv preprint arXiv:2510.10150},
  year={2025}
}
```

## Contact

* Zhezheng Hao — [haozhezheng@zju.edu.cn](mailto:haozhezheng@zju.edu.cn)

-->
