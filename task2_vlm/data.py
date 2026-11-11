"""COCO/VQAv2 manifest construction and deterministic visual-feature caching."""
import json, random
from collections import Counter, defaultdict
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

from task2_vlm.metrics import normalize_answer


def read_json(path):
    with open(path) as stream: return json.load(stream)

def image_path(coco_root, split, image_id):
    return str(Path(coco_root) / split / f"COCO_{split}_{int(image_id):012d}.jpg")

def consensus(annotation):
    """Return an original consensus answer, its agreement count, and all answers."""
    values = [answer["answer"] for answer in annotation["answers"]]
    counts = Counter(normalize_answer(value) for value in values)
    normalized, count = counts.most_common(1)[0]
    # VQAv2 scoring normalizes answers, but the assignment specifies that the
    # training target should be an original answer string from the consensus set.
    target = next(value for value in values if normalize_answer(value) == normalized)
    return target, count, values


def complementary_pairs(questions, annotations, wanted, used_images, seed, *, enforce_disjoint=True, label="VQAv2"):
    """Select same-question VQAv2 pairs without materialising all O(n²) pairs.

    The old implementation built every possible pair for a repeated question.
    Some common questions have thousands of answers, which made preparation take
    effectively forever.  We instead index entries by question/type/answer and
    greedily draw one valid pair per group in successive deterministic passes.
    """
    print(f"Indexing {label} questions and consensus answers...", flush=True)
    answers = {row["question_id"]: row for row in annotations["annotations"]}
    grouped = {"yes/no": defaultdict(lambda: defaultdict(list)), "number": defaultdict(lambda: defaultdict(list)), "other": defaultdict(lambda: defaultdict(list))}
    accepted = 0
    for row in questions["questions"]:
        ann = answers.get(row["question_id"])
        if not ann:
            continue
        target, agreement, all_answers = consensus(ann)
        if agreement < 8:
            continue
        answer_type = ann.get("answer_type", "other")
        kind = answer_type if answer_type in grouped else "other"
        entry = {"image_id": row["image_id"], "question_id": row["question_id"], "question": row["question"], "answer": target, "answers": all_answers, "answer_type": kind}
        grouped[kind][normalize_answer(row["question"])][normalize_answer(target)].append(entry)
        accepted += 1
    print(f"Indexed {accepted:,} high-agreement {label} examples; selecting complementary pairs...", flush=True)

    rng = random.Random(seed)
    quotas = {"yes/no": wanted["yesno"], "number": wanted["number"], "other": wanted["other"]}
    output = []
    selected_pair_ids = set()
    for kind, quota in quotas.items():
        groups = [answers_by_value for answers_by_value in grouped[kind].values() if len(answers_by_value) >= 2]
        rng.shuffle(groups)
        selected = 0
        while selected < quota:
            made_progress = False
            for answers_by_value in groups:
                available_by_answer = {
                    answer: [entry for entry in entries if not enforce_disjoint or entry["image_id"] not in used_images]
                    for answer, entries in answers_by_value.items()
                }
                available_answers = [answer for answer, entries in available_by_answer.items() if entries]
                if len(available_answers) < 2:
                    continue
                left_answer, right_answer = rng.sample(available_answers, 2)
                left, right = rng.choice(available_by_answer[left_answer]), rng.choice(available_by_answer[right_answer])
                if left["image_id"] == right["image_id"]:
                    continue
                pair_id = f"vqa-{left['question_id']}-{right['question_id']}"
                if pair_id in selected_pair_ids:
                    continue
                for side, partner in ((left, right), (right, left)):
                    output.append({**side, "pair_id": pair_id, "partner_image_id": partner["image_id"], "kind": "natural"})
                selected_pair_ids.add(pair_id)
                if enforce_disjoint:
                    used_images.update((left["image_id"], right["image_id"]))
                selected += 1
                made_progress = True
                if selected % 100 == 0 or selected == quota:
                    print(f"  {label} {kind}: {selected}/{quota} pairs", flush=True)
                if selected == quota:
                    break
            if not made_progress:
                raise RuntimeError(f"Insufficient complementary {label} pairs for {kind}: selected {selected}/{quota}.")
    return output


def annotation_pairs(instances, image_ids, wanted_pairs, used_images, seed, *, enforce_disjoint=True, label="COCO"):
    """Create reusable presence/counting pairs using precomputed image indexes."""
    print(f"Indexing {label} object annotations for presence/counting pairs...", flush=True)
    by_image = defaultdict(Counter)
    for ann in instances["annotations"]:
        by_image[ann["image_id"]][ann["category_id"]] += 1
    categories = {row["id"]: row["name"] for row in instances["categories"]}
    category_ids = sorted(categories)
    available = sorted(set(image_ids))

    # Construct these indexes once.  The previous code rebuilt 1--4 count
    # buckets by scanning every COCO image for every requested pair.
    positives = {category: [] for category in category_ids}
    by_count = {category: {count: [] for count in range(1, 5)} for category in category_ids}
    for image_id, counts in by_image.items():
        for category, count in counts.items():
            if category not in positives:
                continue
            positives[category].append(image_id)
            if 1 <= count <= 4:
                by_count[category][count].append(image_id)
    positive_sets = {category: set(ids) for category, ids in positives.items()}
    negatives = {category: [image_id for image_id in available if image_id not in positive_sets[category]] for category in category_ids}

    def available_values(values):
        return values if not enforce_disjoint else [image_id for image_id in values if image_id not in used_images]

    def choose(values):
        candidates = available_values(values)
        return rng.choice(candidates) if candidates else None

    rng = random.Random(seed)
    rows = []
    for kind in ("presence", "counting"):
        selected = 0
        attempts = 0
        while selected < wanted_pairs:
            if attempts >= len(category_ids) * (wanted_pairs + 1):
                raise RuntimeError(f"Insufficient {label} {kind} pairs after image-disjoint filtering.")
            category = category_ids[attempts % len(category_ids)]
            attempts += 1
            if kind == "presence":
                left, right = choose(positives[category]), choose(negatives[category])
                answers = ("yes", "no")
                question = f"Does this image contain {categories[category]}?"
            else:
                counts = [count for count, values in by_count[category].items() if available_values(values)]
                if len(counts) < 2:
                    continue
                first, second = rng.sample(counts, 2)
                left, right = choose(by_count[category][first]), choose(by_count[category][second])
                answers = (str(first), str(second))
                question = f"How many {categories[category]} can you see?"
            if left is None or right is None or left == right:
                continue
            pair_id = f"{kind}-{category}-{left}-{right}"
            rows += [
                {"image_id": left, "partner_image_id": right, "pair_id": pair_id, "question": question, "answer": answers[0], "kind": kind},
                {"image_id": right, "partner_image_id": left, "pair_id": pair_id, "question": question, "answer": answers[1], "kind": kind},
            ]
            if enforce_disjoint:
                used_images.update((left, right))
            selected += 1
            if selected % 100 == 0 or selected == wanted_pairs:
                print(f"  {label} {kind}: {selected}/{wanted_pairs} pairs", flush=True)
    return rows

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2))

def prepare_manifests(cfg):
    coco, vqa, out = Path(cfg["coco_root"]), Path(cfg["vqa_root"]), Path(cfg["prepared_dir"])
    if out.exists(): raise FileExistsError(f"{out} exists; do not silently change selected splits.")
    print("Loading COCO captions and instance annotations...", flush=True)
    cap_train, cap_val = read_json(coco/"annotations/captions_train2014.json"), read_json(coco/"annotations/captions_val2014.json")
    ins_train, ins_val = read_json(coco/"annotations/instances_train2014.json"), read_json(coco/"annotations/instances_val2014.json")
    captions = defaultdict(list)
    for row in cap_train["annotations"]: captions[row["image_id"]].append(row["caption"])
    train_images = sorted(i for i, values in captions.items() if len(values) >= 3)[:6000]
    train_caption = [{"image_id": i, "caption": text, "kind": "caption"} for i in train_images for text in captions[i][:3]]
    val_captions = defaultdict(list)
    for row in cap_val["annotations"]: val_captions[row["image_id"]].append(row["caption"])
    # Evaluation images must be disjoint across both suites and from one
    # another. Training images may recur across curriculum components.
    evaluation_used = set()
    dev_caption_ids = sorted(val_captions)[:96]; evaluation_used.update(dev_caption_ids)
    dev_caption = [{"image_id": i, "caption": val_captions[i][0], "kind": "caption"} for i in dev_caption_ids]
    def vqa_files(split):
        tag = "train2014" if split == "train2014" else "val2014"
        return read_json(vqa/f"v2_OpenEnded_mscoco_{tag}_questions.json"), read_json(vqa/f"v2_mscoco_{tag}_annotations.json")
    print("Loading VQAv2 questions and annotations...", flush=True)
    qtrain, atrain = vqa_files("train2014"); qval, aval = vqa_files("val2014")
    natural_train = complementary_pairs(qtrain, atrain, {"yesno":640,"number":256,"other":640}, set(), cfg["seed"], enforce_disjoint=False, label="training VQAv2")
    natural_dev = complementary_pairs(qval, aval, {"yesno":12,"number":8,"other":12}, evaluation_used, cfg["seed"]+1, label="development VQAv2")
    natural_final = complementary_pairs(qval, aval, {"yesno":36,"number":24,"other":36}, evaluation_used, cfg["seed"]+2, label="final VQAv2")
    # Each call yields `wanted_pairs` presence pairs and `wanted_pairs`
    # counting pairs: 24/64 pairs for development/final as specified.
    annotation_train = annotation_pairs(ins_train, [row["id"] for row in ins_train["images"]], 3000, set(), cfg["seed"]+3, enforce_disjoint=False, label="training COCO")
    annotation_dev = annotation_pairs(ins_val, [row["id"] for row in ins_val["images"]], 12, evaluation_used, cfg["seed"]+4, label="development COCO")
    annotation_final = annotation_pairs(ins_val, [row["id"] for row in ins_val["images"]], 32, evaluation_used, cfg["seed"]+5, label="final COCO")
    for rows, split in ((train_caption,"train2014"),(dev_caption,"val2014"),(natural_train,"train2014"),(natural_dev,"val2014"),(natural_final,"val2014"),(annotation_train,"train2014"),(annotation_dev,"val2014"),(annotation_final,"val2014")):
        for row in rows: row["image_path"] = image_path(coco, split, row["image_id"]); row["split"] = split
    manifest = {"caption_train":train_caption,"caption_dev":dev_caption,"annotation_train":annotation_train,"annotation_dev":annotation_dev,"annotation_final":annotation_final,"natural_train":natural_train,"natural_dev":natural_dev,"natural_final":natural_final,"seed":cfg["seed"]}
    print("Writing deterministic manifest...", flush=True)
    write_json(out/"manifest.json", manifest)
    print(f"Prepared Task 2 manifest at {out}; cache visual features next.", flush=True)


def cache_features(cfg, batch_size=32, scope="all"):
    from transformers import AutoProcessor, SiglipVisionModel
    data = read_json(Path(cfg["prepared_dir"])/"manifest.json")
    scopes = {
        "all": [r for value in data.values() if isinstance(value, list) for r in value],
        "dev": data["caption_dev"] + data["annotation_dev"] + data["natural_dev"],
        "final": data["annotation_final"] + data["natural_final"],
    }
    if scope not in scopes:
        raise ValueError(f"Unknown feature-cache scope: {scope}")
    rows = scopes[scope]
    images = {int(r["image_id"]): r["image_path"] for r in rows}; out = Path(cfg["feature_dir"]); out.mkdir(parents=True, exist_ok=True)
    print(f"Caching {len(images):,} unique {scope} image features...", flush=True)
    processor = AutoProcessor.from_pretrained(cfg["vision_model_id"]); model = SiglipVisionModel.from_pretrained(cfg["vision_model_id"], torch_dtype=torch.float16).cuda().eval()
    ids = [i for i in sorted(images) if not (out/f"{i}.pt").exists()]
    for start in range(0, len(ids), batch_size):
        group = ids[start:start+batch_size]; batch = processor(images=[Image.open(images[i]).convert("RGB") for i in group], return_tensors="pt").pixel_values.cuda()
        with torch.inference_mode(): features = model(pixel_values=batch).last_hidden_state
        if features.shape[1:] != (196, 768): raise RuntimeError(f"Unexpected SigLIP feature shape {tuple(features.shape)}")
        for image_id, feature in zip(group, features): torch.save(feature.cpu(), out/f"{image_id}.pt")
        print(f"Cached {min(start+len(group),len(ids))}/{len(ids)} image features", flush=True)
    del model; torch.cuda.empty_cache()


class ManifestDataset(Dataset):
    def __init__(self, rows, feature_dir): self.rows, self.feature_dir = rows, Path(feature_dir)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = dict(self.rows[index]); row["features"] = torch.load(self.feature_dir/f"{row['image_id']}.pt", map_location="cpu", weights_only=True); return row
