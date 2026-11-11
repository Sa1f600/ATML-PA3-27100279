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
    values = [answer["answer"] for answer in annotation["answers"]]
    counts = Counter(normalize_answer(value) for value in values)
    answer, count = counts.most_common(1)[0]
    return answer, count, values

def complementary_pairs(questions, annotations, wanted, used_images, seed):
    """Select image-disjoint same-question pairs with >=8/10 consensus on both images."""
    answers = {row["question_id"]: row for row in annotations["annotations"]}
    groups = defaultdict(list)
    for row in questions["questions"]:
        ann = answers.get(row["question_id"])
        if not ann: continue
        target, agreement, all_answers = consensus(ann)
        if agreement >= 8: groups[normalize_answer(row["question"])].append({"image_id": row["image_id"], "question_id": row["question_id"], "question": row["question"], "answer": target, "answers": all_answers, "answer_type": ann.get("answer_type", "other")})
    candidates = []
    for question, entries in groups.items():
        for left in entries:
            for right in entries:
                if left["image_id"] < right["image_id"] and left["answer"] != right["answer"]:
                    candidates.append((question, left, right))
    rng = random.Random(seed); rng.shuffle(candidates); output = []
    quotas = {"yes/no": wanted["yesno"], "number": wanted["number"], "other": wanted["other"]}
    for question, left, right in candidates:
        kind = left["answer_type"] if left["answer_type"] in quotas else "other"
        if quotas[kind] <= 0 or left["image_id"] in used_images or right["image_id"] in used_images: continue
        pair_id = f"vqa-{left['question_id']}-{right['question_id']}"
        for side, partner in ((left, right), (right, left)):
            output.append({**side, "pair_id": pair_id, "partner_image_id": partner["image_id"], "kind": "natural"})
        quotas[kind] -= 1; used_images.update((left["image_id"], right["image_id"]))
        if not any(quotas.values()): break
    if any(quotas.values()): raise RuntimeError(f"Insufficient complementary VQA pairs: {quotas}")
    return output

def annotation_pairs(instances, image_ids, wanted_pairs, used_images, seed):
    """Create balanced reusable presence/counting pairs from COCO instance annotations."""
    by_image = defaultdict(Counter)
    for ann in instances["annotations"]: by_image[ann["image_id"]][ann["category_id"]] += 1
    categories = {row["id"]: row["name"] for row in instances["categories"]}; available = sorted(set(image_ids) - set(used_images))
    rng = random.Random(seed); rows = []
    for index in range(wanted_pairs):
        category = sorted(categories)[index % len(categories)]
        positives = [i for i in available if by_image[i][category] > 0]
        negatives = [i for i in available if by_image[i][category] == 0]
        if not positives or not negatives: continue
        yes, no = rng.choice(positives), rng.choice(negatives)
        if yes in used_images or no in used_images: continue
        pair_id = f"presence-{category}-{yes}-{no}"; question = f"Does this image contain {categories[category]}?"
        rows += [{"image_id": yes, "partner_image_id": no, "pair_id": pair_id, "question": question, "answer": "yes", "kind": "presence"}, {"image_id": no, "partner_image_id": yes, "pair_id": pair_id, "question": question, "answer": "no", "kind": "presence"}]
        used_images.update((yes, no))
    for index in range(wanted_pairs):
        category = sorted(categories)[index % len(categories)]
        buckets = {count: [i for i in available if by_image[i][category] == count and i not in used_images] for count in range(1, 5)}
        counts = [count for count, values in buckets.items() if values]
        if len(counts) < 2: continue
        first, second = rng.sample(counts, 2); left, right = rng.choice(buckets[first]), rng.choice(buckets[second])
        pair_id = f"count-{category}-{left}-{right}"; question = f"How many {categories[category]} can you see?"
        rows += [{"image_id": left, "partner_image_id": right, "pair_id": pair_id, "question": question, "answer": str(first), "kind": "counting"}, {"image_id": right, "partner_image_id": left, "pair_id": pair_id, "question": question, "answer": str(second), "kind": "counting"}]
        used_images.update((left, right))
    if len(rows) < 2 * wanted_pairs: raise RuntimeError("Insufficient annotation-derived pairs after image-disjoint filtering.")
    return rows

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2))

def prepare_manifests(cfg):
    coco, vqa, out = Path(cfg["coco_root"]), Path(cfg["vqa_root"]), Path(cfg["prepared_dir"])
    if out.exists(): raise FileExistsError(f"{out} exists; do not silently change selected splits.")
    cap_train, cap_val = read_json(coco/"annotations/captions_train2014.json"), read_json(coco/"annotations/captions_val2014.json")
    ins_train, ins_val = read_json(coco/"annotations/instances_train2014.json"), read_json(coco/"annotations/instances_val2014.json")
    captions = defaultdict(list)
    for row in cap_train["annotations"]: captions[row["image_id"]].append(row["caption"])
    train_images = sorted(i for i, values in captions.items() if len(values) >= 3)[:6000]
    train_caption = [{"image_id": i, "caption": text, "kind": "caption"} for i in train_images for text in captions[i][:3]]
    val_captions = defaultdict(list)
    for row in cap_val["annotations"]: val_captions[row["image_id"]].append(row["caption"])
    used_dev, used_final = set(), set(); dev_caption_ids = [i for i in sorted(val_captions) if i not in used_final][:96]; used_dev.update(dev_caption_ids)
    dev_caption = [{"image_id": i, "caption": val_captions[i][0], "kind": "caption"} for i in dev_caption_ids]
    def vqa_files(split):
        tag = "train2014" if split == "train2014" else "val2014"
        return read_json(vqa/f"v2_OpenEnded_mscoco_{tag}_questions.json"), read_json(vqa/f"v2_mscoco_{tag}_annotations.json")
    qtrain, atrain = vqa_files("train2014"); qval, aval = vqa_files("val2014")
    natural_train = complementary_pairs(qtrain, atrain, {"yesno":640,"number":256,"other":640}, set(train_images), cfg["seed"])
    natural_dev = complementary_pairs(qval, aval, {"yesno":12,"number":8,"other":12}, used_dev, cfg["seed"]+1)
    natural_final = complementary_pairs(qval, aval, {"yesno":36,"number":24,"other":36}, used_dev | used_final, cfg["seed"]+2); used_final.update(row["image_id"] for row in natural_final)
    annotation_train = annotation_pairs(ins_train, [row["id"] for row in ins_train["images"]], 3000, set(train_images), cfg["seed"]+3)
    annotation_dev = annotation_pairs(ins_val, [row["id"] for row in ins_val["images"]], 12, used_dev, cfg["seed"]+4)
    annotation_final = annotation_pairs(ins_val, [row["id"] for row in ins_val["images"]], 32, used_dev | used_final, cfg["seed"]+5)
    for rows, split in ((train_caption,"train2014"),(dev_caption,"val2014"),(natural_train,"train2014"),(natural_dev,"val2014"),(natural_final,"val2014"),(annotation_train,"train2014"),(annotation_dev,"val2014"),(annotation_final,"val2014")):
        for row in rows: row["image_path"] = image_path(coco, split, row["image_id"]); row["split"] = split
    write_json(out/"manifest.json", {"caption_train":train_caption,"caption_dev":dev_caption,"annotation_train":annotation_train,"annotation_dev":annotation_dev,"annotation_final":annotation_final,"natural_train":natural_train,"natural_dev":natural_dev,"natural_final":natural_final,"seed":cfg["seed"]})
    print(f"Prepared Task 2 manifest at {out}; cache visual features next.")


def cache_features(cfg, batch_size=32):
    from transformers import AutoProcessor, SiglipVisionModel
    data = read_json(Path(cfg["prepared_dir"])/"manifest.json"); rows = [r for value in data.values() if isinstance(value, list) for r in value]
    images = {int(r["image_id"]): r["image_path"] for r in rows}; out = Path(cfg["feature_dir"]); out.mkdir(parents=True, exist_ok=True)
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
