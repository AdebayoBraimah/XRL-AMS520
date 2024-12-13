from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from models import GCNFeatureExtractor, FlatFeatureExtractor


class CustomPPOPolicyFlat(ActorCriticPolicy):
    """
    Custom PPO policy with flat feature extractor.
    """

    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicyFlat, self).__init__(
            *args,
            features_extractor_class=FlatFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            **kwargs
        )


class CustomPPOPolicy(ActorCriticPolicy):
    """
    Custom PPO policy integrating the GCN feature extractor.
    """

    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(
            *args,
            features_extractor_class=GCNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            **kwargs
        )
