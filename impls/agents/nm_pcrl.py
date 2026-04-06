from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, StopGradWrapper, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, GCProbabilisticBilinearValue


def history_from_observations_actions(observations, actions):
    prev_actions = jnp.concatenate([jnp.zeros_like(actions[:, :1]), actions[:, :-1]], axis=1)
    history = jnp.concatenate([observations, prev_actions], axis=-1)
    return history


def get_mask(batch_size, context_length):
    blocks = jnp.repeat(jnp.arange(batch_size), context_length)
    mask = jnp.expand_dims(blocks, axis=1) == blocks
    mask = mask - jnp.eye(batch_size * context_length)
    mask = mask * -1e9
    return mask

def logit_mean_sigmoid(logits, axis):
    probs = jax.scipy.special.expit(logits)
    probs = jnp.mean(probs, axis)
    return jax.scipy.special.logit(probs)


class NonMarkovianProbabilisticCRLAgent(flax.struct.PyTreeNode):
    """Non-Markovian Probabilistic Contrastive RL (NM-PCRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def mc_contrastive_loss(self, batch, grad_params, module_name='critic', rng=None):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            actions = batch['actions']
        else:
            actions = None
        v_mean, v_std, phi_dist, psi = self.network.select(module_name)(
            batch['history'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )

        phi = phi_dist.sample(seed=rng, sample_shape=(self.config['pcl_mc_samples'],))

        if len(phi.shape) == 4:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        
        # exclude first context_warmup steps from sequence
        phi = phi[:, :, :, self.config['context_warmup']:]

        # flatten batch_size and context_length
        phi = phi.reshape(phi.shape[0], phi.shape[1], phi.shape[2] * phi.shape[3], phi.shape[4])
        psi = psi.reshape(psi.shape[0], psi.shape[1] * psi.shape[2], psi.shape[3])

        effective_context_length = self.config['context_length'] - self.config['context_warmup']
        
        # get mask to exclude state from same episode as negative goals
        mask = get_mask(batch_size, effective_context_length)
        # add ensemble dimension
        mask = jnp.expand_dims(mask, axis=-1)
        
        logits = jnp.einsum('meik,ejk->ijem', phi, psi) / jnp.sqrt(phi.shape[-1])
        logits = logit_mean_sigmoid(logits, axis=-1)
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size * effective_context_length)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits + mask)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v_mean)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        v_std_shrink = (v_std[:, :, 1:] / v_std[:, :, :-1]).mean()
        v_std_diff = (v_std[:, :, 1:] - v_std[:, :, :-1]).mean()

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'logv_std_mean': v_std.mean(),
            'logv_std_std': v_std.std(),
            'logv_std_shrink': v_std_shrink,
            'logv_std_diff': v_std_diff,
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist, *_ = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist, *_ = self.network.select('actor')(batch['history'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['history'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        batch['history'] = history_from_observations_actions(batch['observations'], batch['actions'])

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.mc_contrastive_loss(batch, grad_params, 'critic', rng=critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['actor_loss'] == 'awr':
            rng, critic_rng = jax.random.split(rng)
            value_loss, value_info = self.mc_contrastive_loss(batch, grad_params, 'value', rng=critic_rng)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        prev_actions=None,
        goals=None,
        seed=None,
        temperature=1.0,
        **kwargs,
    ):
        """Sample actions from the actor."""
        act_obs_pair = jnp.concatenate([observations, prev_actions], axis=-1)
        act_obs_pair = jnp.expand_dims(act_obs_pair, axis=0)
        goals = jnp.expand_dims(goals, axis=(0, 1))
        goals = jnp.repeat(goals, act_obs_pair.shape[1], axis=1)
        dist, *state_info = self.network.select('actor')(act_obs_pair, goals, temperature=temperature, **kwargs)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions[:, -1], *state_info

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            ex_goals: Example batch of goals.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations[:, config['context_warmup']:]
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        history_encoder_module = encoder_modules[config['history_encoder']]
        goal_encoder_module = encoder_modules[config['goal_encoder']]

        encoders['critic_state'] = history_encoder_module()
        if config['common_history_encoder']:
            actor_history_encoder = encoders['critic_state']
        else:
            actor_history_encoder = history_encoder_module()

        if config['freeze_actor_encoder']:
            actor_history_encoder = StopGradWrapper(actor_history_encoder)

        encoders['critic_goal'] = goal_encoder_module()
        encoders['actor'] = GCEncoder(
            state_encoder=actor_history_encoder,
            goal_encoder=goal_encoder_module())
        actor_encoder = encoders['actor']
        if config['actor_loss'] == 'awr':
            encoders['value_state'] = encoder_module()
            encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCProbabilisticBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network to compute advantages (Q - V).
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=False,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        ex_history = history_from_observations_actions(ex_observations, ex_actions)

        network_info = dict(
            critic=(critic_def, (ex_history, ex_goals, ex_actions)),
            actor=(actor_def, (ex_history, ex_goals)),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='nm_pcrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=64,  # Batch size.
            actor_hidden_dims=(512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            history_encoder='gru',
            goal_encoder='none',
            # Dataset hyperparameters.
            dataset_class='GCPODataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            context_length=16,
            context_warmup=0,
            eval_context_type='full', # 'full', 'trunc'
            occlusion=dict(
                type='none',
                drop_prob=ml_collections.config_dict.placeholder(float),
                noise_std=ml_collections.config_dict.placeholder(float),
                occlude_goals=False,
            ),
            common_history_encoder=False,
            freeze_actor_encoder=False,
            pcl_mc_samples=16,
        )
    )
    return config
