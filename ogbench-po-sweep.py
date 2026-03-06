Product([
  Singleton({
    'eval_episodes': 50,
  }),
  # agents
  Concat([
    # non-markovian crl
    Product([
      Singleton({
        'agent': 'agents/nm_crl.py',
        'agent.lr': 0.0001,
        'agent.common_history_encoder': False,
        'agent.freeze_actor_encoder': False,
        'agent.eval_context_type': 'trunc',
      }),
      Parallel({
        'agent.context_length': [4, 16],
        'agent.batch_size': [256, 64],
      }),
    ]),
    # baseline crl
    Singleton({
      'agent': 'agents/crl.py',
      'agent.lr': 0.0003,
      'agent.batch_size': 1024,
    }),
  ]),
  # environments, hyperparameters as per ogbench
  Concat([
    Singleton({
      'env_name': 'antmaze-medium-navigate-v0',
      'agent.alpha': 0.1,
    }),
    Singleton({
      'env_name': 'antmaze-medium-stitch-v0',
      'agent.actor_p_randomgoal': 0.5,
      'agent.actor_p_trajgoal': 0.5,
      'agent.alpha': 0.1,
    }),
    Singleton({
      'env_name': 'humanoidmaze-medium-navigate-v0',
      'agent.alpha': 0.1,
      'agent.discount': 0.995,
    }),
    Singleton({
      'env_name': 'cube-single-play-v0',
      'agent.alpha': 3.0,
    }),
  ]),
  # occlusions
  Concat([
    Product({
      'agent.occlusion.type': ['noise'],
      'agent.occlusion.noise_std': [0.2, 0.4],
    }),
    Product({
      'agent.occlusion.type': ['flickering'],
      'agent.occlusion.drop_prob': [0.2, 0.5],
    }),
  ]),
  # seeds
  Parallel({
    'seed': range(0, 5),
  }),
])
