"""Mengubah pohon keputusan menjadi data tata letak untuk digambar sebagai SVG di template."""
import numpy as np

from config import CLASS_NAMES, FEATURES, LABELS
from train_model import format_number

NODE_W, NODE_H = 112, 76
SLOT_W, LEVEL_H = 124, 120
PAD_X, PAD_Y = 6, 8
TONE_LEVELS = 4


def build_tree_view(tree, stats):
    """Hitung posisi simpul dan garis penghubung. Hasilnya dipakai macro tree_svg."""
    structure = tree.tree_
    nodes, edges = {}, []
    leaf_count = [0]

    def place(node, depth):
        left, right = structure.children_left[node], structure.children_right[node]
        is_leaf = left == -1
        if is_leaf:
            center = PAD_X + leaf_count[0] * SLOT_W + SLOT_W / 2
            leaf_count[0] += 1
        else:
            center = (place(left, depth + 1) + place(right, depth + 1)) / 2
            for child, label in ((left, "Ya"), (right, "Tidak")):
                edges.append({"parent": int(node), "child": int(child), "label": label})
        nodes[node] = _describe_node(structure, node, depth, center, stats[node], is_leaf)
        return center

    place(0, 0)
    depth = max(n["depth"] for n in nodes.values())
    _attach_edge_geometry(edges, nodes)
    return {
        "nodes": [nodes[i] for i in sorted(nodes)],
        "edges": edges,
        "width": int(PAD_X * 2 + leaf_count[0] * SLOT_W),
        "height": int(PAD_Y * 2 + depth * LEVEL_H + NODE_H),
        "node_w": NODE_W,
        "node_h": NODE_H,
    }


def _describe_node(structure, node, depth, center, stats, is_leaf):
    weighted = structure.value[node][0]
    share = weighted / weighted.sum()
    predicted = int(np.argmax(share))
    level = 1 + min(TONE_LEVELS - 1, int(abs(share[1] - share[0]) * TONE_LEVELS))
    share_positive = stats["pos"] / stats["n"] if stats["n"] else 0.0

    lines = [f"n = {stats['n']}",
             f"diabetes {share_positive * 100:.0f}%",
             CLASS_NAMES[predicted]]
    classes = ["t-n", "t-share", "t-pred"]
    if not is_leaf:
        name = LABELS[FEATURES[structure.feature[node]]]
        lines.insert(0, f"{name} \u2264 {format_number(structure.threshold[node])}")
        classes.insert(0, "t-question")

    top = 17 if not is_leaf else 26
    text = [{"text": t, "cls": c, "y": round(top + i * 17.5, 1)}
            for i, (t, c) in enumerate(zip(lines, classes))]
    return {
        "id": int(node),
        "depth": depth,
        "x": round(center - NODE_W / 2, 1),
        "y": PAD_Y + depth * LEVEL_H,
        "center": center,
        "tone": f"tone-{'pos' if predicted == 1 else 'neg'}-{level}",
        "is_leaf": is_leaf,
        "lines": text,
    }


def _attach_edge_geometry(edges, nodes):
    for edge in edges:
        parent, child = nodes[edge["parent"]], nodes[edge["child"]]
        x1, y1 = parent["center"], parent["y"] + NODE_H
        x2, y2 = child["center"], child["y"]
        mid = (y1 + y2) / 2
        edge["d"] = f"M{x1:.1f} {y1} C{x1:.1f} {mid:.1f} {x2:.1f} {mid:.1f} {x2:.1f} {y2}"
        side = -1 if edge["label"] == "Ya" else 1
        edge["label_x"] = round((x1 + x2) / 2 + side * 14, 1)
        edge["label_y"] = round(mid - 4, 1)
