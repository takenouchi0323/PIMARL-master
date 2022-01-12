## PIMARL: Permutation Invariant for Multi-Agent Reinforcement Learning #


#### 必要なライブラリおよび動作を確認したバージョン：
* Python 3.9.7
* Pytorch 1.9.0+cu102 (1.10ではRuntimeErrorが発生します)
* OpenAI gym 0.10.9 (https://github.com/openai/gym)
* numba 0.54.1
* bridson 0.1.0
* ddpg 0.2.0


ddpgのライブラリのフォルダ内にある'ddpg/__init__.py'ファイルをソースコード内の'ddpg/__init__.py'ファイルに置き換える．
（置き換えないと'simple_tag'シナリオでエラーが発生します．）

### インストール：

MPE環境のインストール

    cd multiagent-particle-envs
    pip install -e .


### 学習：

config_deepsetファイルなどに設定を保存して学習を始める．

    cd maddpg
    python main.py deepset
    
### 学習済みデータからの実行：

config_playファイルに設定を保存して実行を始める．設定する各種パラメータは学習時に書き出したconfigファイルに記録してあるので，基本的にはその値を用いる．

    cd maddpg
    python play.py play


### Acknowledgement
The MADDPG code is based on the DDPG implementation of https://github.com/ikostrikov/pytorch-ddpg-naf

The improved MPE code is based on the MPE implementation of https://github.com/openai/multiagent-particle-envs

The GCN code is based on the implementation of https://github.com/tkipf/gcn

The GCN code of PIMARL is based on the implementation of https://github.com/IouJenLiu/PIC

The DeepSets code of PIMARL is based on the implementation of https://arxiv.org/pdf/2105.08268.pdf
