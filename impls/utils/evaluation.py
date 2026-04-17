from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    nm_agent = config['agent_name'].startswith('nm_')

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))

        if nm_agent:
            cache = dict(carry=None)
        else:
            cache = dict()

        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            kwargs = dict()
            if nm_agent:
                if config['eval_context_type'] == 'trunc':
                    kwargs['prev_actions'] = np.stack(
                        [np.zeros(env.action_space.shape)]
                        + traj['action'][-config['context_length']:],
                        axis=0)
                    kwargs['observations'] = np.stack(
                        traj['observation'][-config['context_length']:]
                        + [observation],
                        axis=0)
                elif config['eval_context_type'] == 'full' and len(traj['action']) > 0:
                    kwargs['prev_actions'] = np.expand_dims(traj['action'][-1], axis=0)
                    kwargs['observations'] = np.expand_dims(observation, axis=0)
                    kwargs['carry'] = cache.get('carry')
                else:
                    kwargs['prev_actions'] = np.zeros((1, *env.action_space.shape))
                    kwargs['observations'] = np.expand_dims(observation, axis=0)

            else:
                kwargs['observations'] = observation

            action, *state_info = actor_fn(goals=goal, temperature=eval_temperature, **kwargs)
            action = np.reshape(action, env.action_space.shape)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            if nm_agent and len(state_info) > 0:
                cache['carry'] = state_info[0].get('carry')

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    stats['success'] = np.sort(stats['success'])

    for p in [2, 5, 10, 25]:
        stats[f'success_{p}_percentile'] = np.percentile(stats['success'], p)
        stats[f'success_{p}_percentile_mean'] = stats['success'][:int(np.ceil(p / 100 * stats['success'].size))].mean()

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
