from prefix_training.dataset import PrefixBank, PrefixEpisodes
from . import spec


class FourPrefixBank(PrefixBank):
    spec_module = spec


class FourPrefixEpisodes(PrefixEpisodes):
    def __init__(self, sources, horizon=10):
        super().__init__(sources, horizon, kind='vla_prefix4_tail15',
                         bank_class=FourPrefixBank, segment_count=4,
                         dataset_policy='a60_plus_prefix4_tail15_v1')
