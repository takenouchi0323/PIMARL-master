from util.configdict import ConfigDict


def get_config():

    config = ConfigDict()
    config.alg = ConfigDict()
    config.nn = ConfigDict()
    config.env = ConfigDict()
    config.main = ConfigDict()

    #Actor-Networkについての項目
    config.alg.actor_type = 'mlp'  #* 'mlp', 'gcn_max', 'deepset'
    #config.alg.use_allobs = False #actor_typeがmlpのときに全エージェントの状態を入力とするかどうか
    #config.alg.actor_lr = 1e-2 #学習率
    #config.alg.actor_updates_per_step = 100 #更新の頻度
    #config.alg.steps_per_actor_update = 100 #更新の数
    config.nn.actor_hidden_size = 187 #隠れ層のノード数
    
    #Critic-Networkについての項目
    config.alg.critic_type = 'gcn_max'  #* 'mlp', 'gcn_max', 'deepset', 'deepset2'
    #config.alg.critic_lr = 0.00008 #* 学習率，シナリオによる
    #config.alg.critic_updates_per_step = 100 #更新の頻度
    #config.alg.steps_per_critic_update = 100 #更新の数
    config.nn.critic_hidden_size = 187 #隠れ層のノード数

    #config.alg.batch_size = 32 #バッチサイズ
    config.alg.replay_size = 800 #バッファーのサイズ
    #config.alg.target_update_mode = 'soft'  # 'soft', 'hard' Critic-Networkの更新方法
    config.alg.use_buffer = False #* Actor-Networkに対して行動決定時にバッファーを使うかどうか
    #config.alg.noise = 50 #* 0~100 学習時のActor-Networkへのノイズの大きさ，0でノイズなし
    config.alg.eval_buffer = False #* バッファーの各観測値を評価するかどうか

    #config.alg.train_noise = False #行動決定時のノイズを学習時に入れるかどうか
    #config.alg.fixed_lr = False #学習率を途中で調整するかどうか
    config.alg.gamma = 0.95 #割引率
    #config.alg.tau = 0.01 #ソフトアップデートの割合
    
    config.env.scenario = 'simple_spread2_n3'#* 'simple_spread_n?', 'simple_spread2_n?', 'simple_push_n?', 'simple_tag_n?'
    config.env.num_steps = 25 #* 1エピソードでのステップ数（simple_tagのときは50，それ以外は25）
    config.env.num_episodes = 60000 #* 学習するエピソードの数（simple_tagのときは30000，それ以外は60000）
    config.env.eval_freq = 1000  #ログを出力するエピソード間隔
    config.env.num_eval_runs = 1000 #評価時のエピソード数
    
    #config.main.cuda = True
    #config.main.cuda_num = 'cuda:0'
    config.main.exp_name = 'coop_spread2_n3_gcn_000008'
    config.main.save_dir = './ckpt_plot_spread2'
    config.main.seed = 12340
    config.main.render_episodes = None #Trueなどで描画

    # Use /util/count_params to calculate the number of hidden layer units
    # needed for equivalent number of params as PIC with 128 units
    # Use 185/187/187/187 for N=3/6/15/30, respectively
    #config.nn.hidden_size = 187

    return config