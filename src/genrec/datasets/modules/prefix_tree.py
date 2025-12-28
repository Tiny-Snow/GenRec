"""Prefix tree helpers used for constrained decoding with SID tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "PrefixTree",
]


@dataclass
class PrefixTreeNode:
    """Node in a prefix tree storing SID token sequences."""

    children: Dict[int, "PrefixTreeNode"] = field(default_factory=dict)
    is_terminal: bool = False
    item_id: Optional[int] = None


class PrefixTree:
    """Prefix tree that constrains decoding to catalogue-backed candidates.

    The tree stores discrete SID token sequences and the item IDs they map to.
    All operations are intentionally allocation-free except for returning new
    `list` objects to callers, keeping inference-time overhead low.
    """

    def __init__(self) -> None:
        self._root = PrefixTreeNode()

    @classmethod
    def from_mapping(cls, mapping: Dict[int, Sequence[int]]) -> "PrefixTree":
        """Builds a tree from a mapping of item IDs to SID sequences."""

        tree = cls()
        for item_id, tokens in mapping.items():
            tree.insert(tokens, item_id)
        return tree

    def insert(self, tokens: Sequence[int], item_id: int) -> None:
        """Stores `tokens` in the tree and associates them with `item_id`."""

        node = self._root
        for token in tokens:
            node = node.children.setdefault(int(token), PrefixTreeNode())
        node.is_terminal = True
        node.item_id = int(item_id)

    def is_valid_prefix(self, prefix: Sequence[int]) -> bool:
        """Returns `True` if `prefix` matches at least one catalogue path."""

        return self._descend(prefix) is not None

    def is_terminal(self, prefix: Sequence[int]) -> bool:
        """Returns `True` if `prefix` already corresponds to a full item."""

        node = self._descend(prefix)
        return bool(node and node.is_terminal)

    def next_tokens(self, prefix: Sequence[int]) -> List[int]:
        """Lists candidate next tokens that keep the prefix valid."""

        node = self._descend(prefix)
        if node is None:
            return []
        return list(node.children.keys())

    def iter_completions(self, prefix: Sequence[int]) -> Iterator[Tuple[List[int], int]]:
        """Yields completions `(tokens, item_id)` compatible with `prefix`."""

        node = self._descend(prefix)
        if node is None:
            return iter(())
        return self._iter_from_node(prefix, node)

    def _descend(self, prefix: Sequence[int]) -> Optional[PrefixTreeNode]:
        """Traverses the tree according to `prefix` and returns the end node."""

        node = self._root
        for token in prefix:
            node = node.children.get(int(token))
            if node is None:
                return None
        return node

    def _iter_from_node(self, prefix: Sequence[int], node: PrefixTreeNode) -> Iterator[Tuple[List[int], int]]:
        """Depth-first iterator yielding all completions under `node`."""

        if node.is_terminal and node.item_id is not None:
            yield list(prefix), node.item_id
        for token, child in node.children.items():
            next_prefix = list(prefix) + [token]
            yield from self._iter_from_node(next_prefix, child)
