import pytest

from genrec.datasets.modules.prefix_tree import PrefixTree


@pytest.fixture()
def sample_mapping():
    return {
        101: [1, 2, 3],
        102: [1, 2, 4],
        103: [1, 5],
        104: [2, 6],
    }


def test_prefix_tree_basic_operations(sample_mapping):
    tree = PrefixTree.from_mapping(sample_mapping)

    assert tree.is_valid_prefix([])
    assert tree.is_valid_prefix([1])
    assert tree.is_valid_prefix([1, 2])
    assert not tree.is_valid_prefix([9])

    assert tree.is_terminal([1, 2, 3])
    assert tree.is_terminal([1, 2, 4])
    assert tree.is_terminal([1, 5])
    assert not tree.is_terminal([1, 2])

    next_tokens = tree.next_tokens([1, 2])
    assert set(next_tokens) == {3, 4}
    assert tree.next_tokens([9]) == []

    completions = list(tree.iter_completions([1, 2]))
    assert sorted(completions) == [([1, 2, 3], 101), ([1, 2, 4], 102)]

    assert list(tree.iter_completions([3])) == []


def test_prefix_tree_insert_updates_item_ids(sample_mapping):
    tree = PrefixTree.from_mapping(sample_mapping)

    tree.insert([1, 2, 3], item_id=999)
    assert tree.is_terminal([1, 2, 3])
    assert list(tree.iter_completions([1, 2, 3])) == [([1, 2, 3], 999)]

    tree.insert([7, 8], item_id=105)
    assert tree.is_valid_prefix([7])
    assert tree.is_terminal([7, 8])
    assert tree.next_tokens([7]) == [8]


def test_prefix_tree_iter_completions_returns_independent_lists(sample_mapping):
    tree = PrefixTree.from_mapping(sample_mapping)

    completions = list(tree.iter_completions([1]))
    for prefix, _ in completions:
        prefix.append(99)

    fresh = list(tree.iter_completions([1]))
    for prefix, _ in fresh:
        assert prefix[-1] != 99
