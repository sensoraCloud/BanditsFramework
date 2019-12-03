class TreeSegmentParameters:
    def __init__(self, actions, min_sample_conf, segment_num_trees, tree_max_depth, tree_conf_bound, reward_agg_func):
        self.actions = actions
        self.min_sample_conf = min_sample_conf
        self.segment_num_trees = segment_num_trees
        self.tree_max_depth = tree_max_depth
        self.tree_conf_bound = tree_conf_bound
        self.reward_agg_func = reward_agg_func
