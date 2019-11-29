from collections import deque

import numpy as np
import pandas as pd

from voucher_opt.transformers.segmentation.tree_segment_transformer import TreeSegmentationBuilder, \
    TreeSegmentParameters


def assert_complete(node):
    if node.left and node.right:
        assert_complete(node.left)
        assert_complete(node.right)
    else:
        assert node.left is None, 'Tree is not complete.'
        assert node.right is None, 'Tree is not complete.'


def assert_depth(node, depth):
    d = 0
    q = deque([node])
    while q:
        node = q.popleft()
        if node.depth > d:
            d = node.depth
        if node.left is None and node.right is None:
            continue
        q.append(node.left)
        q.append(node.right)
    assert d == depth, f'Expected tree depth = {depth}, but actual tree depth is {d}'


def test_build_tree_no_confident_features():
    universe_df = pd.DataFrame([
        [1, 0, 1, 0.01],
        [2, 1, 1, 0.01],
        [3, 0, 1, 0.01],
        [1, 1, 0, 0.5],
        [2, 0, 0, 0.5],
        [3, 1, 0, 0.5],
    ], columns=['action_code', 'has_sent_freebie', 'is_linux_user', 'norm_feedback'])
    builder = TreeSegmentationBuilder(universe_df, TreeSegmentParameters([1, 2, 3], 3, 10, 6, 0.1, np.mean))
    tree = builder.build_tree()
    assert_complete(tree)
    assert tree.left is None
    assert tree.right is None
    assert_depth(tree, 1)


def test_build_tree_1_confident_feature():
    universe_df = pd.DataFrame([
        [1, 0, 1, 0.01],
        [2, 1, 1, 0.01],
        [3, 0, 1, 0.01],
        [1, 1, 0, 0.5],
        [2, 0, 0, 0.5],
        [3, 1, 0, 0.5],
    ], columns=['action_code', 'has_sent_freebie', 'is_linux_user', 'norm_feedback'])
    builder = TreeSegmentationBuilder(universe_df, TreeSegmentParameters([1, 2, 3], 1, 10, 6, 0.1, np.mean))
    tree = builder.build_tree()
    assert_complete(tree)
    assert_depth(tree, 2)


def test_build_tree_2_confident_features():
    universe_df = pd.DataFrame([
        [1, 0, 0, 0.01],
        [1, 0, 1, 0.2],
        [1, 0, 1, 0.9],
        [1, 1, 0, 0.5],
        [1, 1, 1, 0.6],
        [2, 0, 0, 0.9],
        [2, 0, 1, 0.01],
        [2, 1, 0, 0.01],
        [2, 1, 1, 0.01],
    ], columns=['action_code', 'has_sent_freebie', 'is_linux_user', 'norm_feedback'])
    builder = TreeSegmentationBuilder(universe_df, TreeSegmentParameters([1, 2], 1, 10, 6, 0.01, np.mean))
    tree = builder.build_tree()
    assert_complete(tree)
    assert_depth(tree, 3)

# TODO: Add tests for Experian features, test that split on has_experian is enforced
