import numpy as np


class BaseOcclusion:
    """
    Base class for occlusion transforms.
    """
    def __init__(self, obs_ndim, config):
        self.obs_ndim = obs_ndim

    def __call__(self, observation):
        raise NotImplementedError


class NoOcclusion(BaseOcclusion):
    """
    Default, no occlusion.
    """
    def __call__(self, observation):
        return observation


class Flickering(BaseOcclusion):
    """
    Flicker or drop (zero out) the state vector randomly with probability drop_prob.
    Attributes:
        drop_prob: probability of dropping the state
    """
    def __init__(self, obs_ndim, config):
        super().__init__(obs_ndim, config)
        self.drop_prob = config['drop_prob']

    def __call__(self, observation):
        batch_shape = observation.shape[:-self.obs_ndim]
        # Bernoulli samples for random dropping
        drop_mask = np.random.random(size=batch_shape) > self.drop_prob
        drop_mask = np.expand_dims(drop_mask, axis=tuple(len(batch_shape) + np.arange(self.obs_ndim)))
        observation = observation * drop_mask
        return observation


class Noise(BaseOcclusion):
    """
    Add zero-mean normal noise.
    Attributes:
        noise_std: standard deviation of additive noise
    """
    def __init__(self, obs_ndim, config):
        super().__init__(obs_ndim, config)
        self.noise_std = config['noise_std']

    def __call__(self, observation):
        noise = np.random.normal(scale=self.noise_std, size=observation.shape)
        observation = observation + noise
        return observation


class SensorDropout(BaseOcclusion):
    """
    Drop (zero out) the state vector components iid randomly with probability drop_prob.
    Attributes:
        drop_prob: probability of dropping a state component
    """
    def __init__(self, obs_ndim, config):
        super().__init__(obs_ndim, config)
        self.drop_prob = config['drop_prob']

    def __call__(self, observation):
        # Bernoulli samples for random dropping
        drop_mask = np.random.random(size=observation.shape) > self.drop_prob
        observation = observation * drop_mask
        return observation


class VelocityOcclusion(BaseOcclusion):
    """
    """
    def __init__(self, obs_ndim, config):
        super().__init__(obs_ndim, config)

    def __call__(self, observation):
        raise NotImplementedError


def make_occlusion_module(obs_ndim, config):
    module = {
        'none': NoOcclusion,
        'flickering': Flickering,
        'noise': Noise,
        'sensor_dropout': SensorDropout,
    }.get(config['type'])
    return module(obs_ndim, config)
