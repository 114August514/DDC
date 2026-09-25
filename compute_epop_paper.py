"""Paper popularity direction for a trained LightGCN checkpoint.

arXiv:2512.10688v6, Equation (4):

    e_pop = normalize(mean(e_i | i in I_head) - mean(e_i | i in I_tail))

I_head and I_tail are the highest and lowest training-frequency items,
each a fraction rho of the catalog. The paper's example is rho = 0.05.
Item embeddings are the vectors used in the score, LightGCN.forward()'s
final item embeddings. Pass --embedding ego to use the layer-0 table instead.

PCA is not used. After normalization, the direction is flipped when the
head centroid projects below the tail centroid, so head · e_pop > tail · e_pop.

The written JSON keeps the released loader contract: entry [1]["value"]["0"]
is a length-1 list of the direction. lightgcnddc.py reads that entry and
L2-normalizes it again.

    uv run python compute_epop_paper.py --self-test
    uv run python compute_epop_paper.py \\
        --model_file ./saved/LightGCN-Sep-22-2026_17-41-29.pth \\
        --output ./e_pop_saved/tmall/e_pop_paper.json
"""

import argparse
import json
import os
import sys

# Set the visible GPU before torch initializes CUDA.
if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu_id", type=int, default=0)
    _pre.add_argument("--self-test", action="store_true")
    _known, _ = _pre.parse_known_args(sys.argv[1:])
    if _known.self_test:
        pass
    elif _known.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL_FILE = "./saved/LightGCN-Sep-22-2026_17-41-29.pth"
DEFAULT_OUTPUT = "./e_pop_saved/tmall/e_pop_paper.json"
DEFAULT_COMPARE = "./e_pop_saved/tmall/rep_direction_item_tmall_20250802_203218.json"


def select_head_tail(counts, rho):
    """Return head and tail item ids.

    counts[0] is the padding slot and is never selected. Ranking is
    training-interaction count descending, item id ascending on ties.
    k = floor(rho * n_real).
    """
    if counts.ndim != 1:
        raise ValueError("counts must be a 1-D tensor with padding at index 0")
    if not (0.0 < rho < 0.5):
        raise ValueError("rho must be in (0, 0.5) so head and tail are disjoint")

    n_real = counts.numel() - 1
    k = int(rho * n_real)
    if k < 1:
        raise ValueError(f"rho={rho} selects no items from {n_real} real items")
    if 2 * k > n_real:
        raise ValueError(
            f"rho={rho} selects k={k} twice from {n_real} items; head and tail overlap"
        )

    real_ids = np.arange(1, counts.numel())
    real_counts = counts[1:].detach().cpu().numpy()
    order = np.lexsort((real_ids, -real_counts))
    ordered_ids = real_ids[order]
    head_ids = torch.from_numpy(ordered_ids[:k].copy()).long()
    tail_ids = torch.from_numpy(ordered_ids[-k:].copy()).long()
    return head_ids, tail_ids


def align_popularity_sign(direction, head_embeddings, tail_embeddings):
    """Flip direction when the head centroid lies below the tail centroid."""
    direction = F.normalize(direction.float(), p=2, dim=0)
    head_proj = torch.dot(head_embeddings.float().mean(dim=0), direction)
    tail_proj = torch.dot(tail_embeddings.float().mean(dim=0), direction)
    flipped = bool(head_proj < tail_proj)
    if flipped:
        direction = -direction
        head_proj = -head_proj
        tail_proj = -tail_proj
    return direction, flipped, float(head_proj), float(tail_proj)


def paper_epop(embeddings, head_ids, tail_ids):
    """Unit popularity direction from Equation (4), then the sign check."""
    head_embeddings = embeddings.index_select(0, head_ids.to(embeddings.device))
    tail_embeddings = embeddings.index_select(0, tail_ids.to(embeddings.device))
    diff = head_embeddings.float().mean(dim=0) - tail_embeddings.float().mean(dim=0)
    distance = torch.linalg.vector_norm(diff)
    if float(distance) < 1e-12:
        raise ValueError("head and tail centroids coincide; e_pop is undefined")
    direction, flipped, head_proj, tail_proj = align_popularity_sign(
        diff, head_embeddings, tail_embeddings
    )
    return direction, flipped, float(distance), head_proj, tail_proj


def direction_record(direction):
    """Shape expected by lightgcnddc.py: data[1]['value']['0'] then squeeze(0)."""
    vector = direction.detach().cpu().float().tolist()
    return {"0": [vector]}


def load_direction_record(payload):
    raw = payload[1]["value"]["0"]
    return torch.tensor(raw, dtype=torch.float32).squeeze(0)


def build_payload(direction, config_meta, diagnostics):
    return [
        {"key": "paper_epop_config", "value": config_meta},
        {"key": "paper_epop", "value": direction_record(direction)},
        {"key": "paper_epop_diagnostics", "value": diagnostics},
    ]


def training_item_counts(train_dataset):
    """Interaction frequency on the training split. Pop(i) = |U_i^+|."""
    item_ids = train_dataset.inter_feat[train_dataset.iid_field]
    if not isinstance(item_ids, torch.Tensor):
        item_ids = torch.tensor(item_ids)
    return torch.bincount(item_ids.cpu().long(), minlength=train_dataset.item_num)


def final_item_embeddings(model):
    model.eval()
    with torch.no_grad():
        _user_embeddings, item_embeddings = model.forward()
    return item_embeddings.detach()


def ego_item_embeddings(model):
    return model.item_embedding.weight.detach()


def cosine_with_released(direction, path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        released = load_direction_record(json.load(handle))
    released = F.normalize(released.float(), p=2, dim=0)
    return float(torch.dot(direction.cpu().float(), released))


def compute_from_checkpoint(args):
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, init_seed

    checkpoint = torch.load(args.model_file, map_location="cpu")
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config["gpu_id"] = 0 if device.type == "cuda" else -1
    config["device"] = device

    dataset = create_dataset(config)
    train_data, _valid_data, _test_data = data_preparation(config, dataset)
    train_dataset = train_data.dataset

    model = get_model(config["model"])(config, train_dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    if args.embedding == "final":
        item_embeddings = final_item_embeddings(model)
    elif args.embedding == "ego":
        item_embeddings = ego_item_embeddings(model)
    else:
        raise ValueError(f"unknown embedding source: {args.embedding}")

    counts = training_item_counts(train_dataset)
    if counts.numel() != item_embeddings.size(0):
        raise ValueError(
            "item count length "
            f"{counts.numel()} does not match embeddings {tuple(item_embeddings.shape)}"
        )

    head_ids, tail_ids = select_head_tail(counts, args.rho)
    direction, flipped, distance, head_proj, tail_proj = paper_epop(
        item_embeddings, head_ids, tail_ids
    )
    released_cosine = cosine_with_released(direction, args.compare_to)

    head_counts = counts[head_ids]
    tail_counts = counts[tail_ids]
    config_meta = {
        "formula": "normalize(mean(head) - mean(tail))",
        "rho": args.rho,
        "k": int(head_ids.numel()),
        "embedding": args.embedding,
        "model_file": args.model_file,
        "dataset": config["dataset"],
        "seed": config["seed"],
        "n_layers": config["n_layers"],
        "embedding_size": config["embedding_size"],
        "compare_to": args.compare_to,
    }
    diagnostics = {
        "n_real_items": int(counts.numel() - 1),
        "sign_flipped": flipped,
        "centroid_l2": distance,
        "head_projection": head_proj,
        "tail_projection": tail_proj,
        "head_count_min": int(head_counts.min()),
        "head_count_max": int(head_counts.max()),
        "tail_count_min": int(tail_counts.min()),
        "tail_count_max": int(tail_counts.max()),
        "cosine_with_released": released_cosine,
        "direction_l2": float(torch.linalg.vector_norm(direction.cpu())),
        "head_ids": head_ids.tolist(),
        "tail_ids": tail_ids.tolist(),
    }
    return build_payload(direction.cpu(), config_meta, diagnostics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute the paper centroid popularity direction e_pop."
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model_file", type=str, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="Head and tail fraction. The paper's example is 0.05.",
    )
    parser.add_argument(
        "--embedding",
        choices=["final", "ego"],
        default="final",
        help="final: LightGCN propagated embeddings used in the score. ego: layer-0 table.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Negative value forces CPU.")
    parser.add_argument(
        "--compare_to",
        type=str,
        default=DEFAULT_COMPARE,
        help="Released direction JSON. Cosine is stored and the file is not modified.",
    )
    return parser.parse_args(argv)


def run_self_test():
    counts = torch.tensor([1000, 1, 1, 5, 9, 9, 2])
    head_ids, tail_ids = select_head_tail(counts, rho=1.0 / 3.0)
    if head_ids.tolist() != [4, 5]:
        raise AssertionError(f"head ids {head_ids.tolist()}")
    if tail_ids.tolist() != [1, 2]:
        raise AssertionError(f"tail ids {tail_ids.tolist()}")
    if 0 in head_ids.tolist() or 0 in tail_ids.tolist():
        raise AssertionError("padding id was selected")

    pad = select_head_tail(torch.tensor([1000, 1, 2, 3, 4]), rho=0.25)
    if pad[0].tolist() != [4] or pad[1].tolist() != [1]:
        raise AssertionError(f"padding exclusion failed: {pad[0].tolist()} {pad[1].tolist()}")

    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [-2.0, 0.0],
            [-4.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [0.0, -1.0],
        ]
    )
    direction, flipped, distance, head_proj, tail_proj = paper_epop(
        embeddings, head_ids, tail_ids
    )
    expected = torch.tensor([1.0, 0.0])
    if flipped:
        raise AssertionError("centroid difference was flipped")
    if not torch.allclose(direction, expected, atol=1e-6):
        raise AssertionError(f"direction {direction.tolist()}")
    if not (head_proj > tail_proj):
        raise AssertionError(f"projections {head_proj} {tail_proj}")
    if abs(distance - 6.0) > 1e-5:
        raise AssertionError(f"centroid distance {distance}")

    reversed_direction, reversed_flipped, reversed_head, reversed_tail = (
        align_popularity_sign(-expected, embeddings[head_ids], embeddings[tail_ids])
    )
    if not reversed_flipped or not torch.allclose(reversed_direction, expected, atol=1e-6):
        raise AssertionError("sign alignment did not restore head > tail")
    if not (reversed_head > reversed_tail):
        raise AssertionError("flipped projections are still reversed")

    payload = build_payload(
        direction,
        {"rho": 1.0 / 3.0, "k": 2},
        {"sign_flipped": False},
    )
    loaded = load_direction_record(payload)
    if loaded.shape != (2,) or not torch.allclose(loaded, expected, atol=1e-6):
        raise AssertionError(f"json contract failed: {loaded}")
    if payload[1]["key"] != "paper_epop":
        raise AssertionError("direction must stay at payload index 1")

    try:
        select_head_tail(counts, rho=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("rho=0.5 should be rejected")

    print("self-test ok")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return

    payload = compute_from_checkpoint(args)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    diagnostics = payload[2]["value"]
    print(f"wrote {output_path}")
    print(
        "k={k} sign_flipped={sign_flipped} head_proj={head_projection:.6f} "
        "tail_proj={tail_projection:.6f} cosine_with_released={cosine_with_released}".format(
            k=payload[0]["value"]["k"],
            **diagnostics,
        )
    )


if __name__ == "__main__":
    main()
