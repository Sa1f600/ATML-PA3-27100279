"""Final four-image-condition grounding evaluation for Task 2."""
import argparse, json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.experiment import read_json, write_json
from task2_vlm.data import read_json as read_manifest
from task2_vlm.metrics import bootstrap_pair_difference, exact_score, vqa_soft_score
from task2_vlm.model import Connector, VLM, install_lora
from task2_vlm.train import _cap_tokens, _find_subsequence, render

CONDITIONS=('correct','swap','mismatch','no_image')

def load_vlm(cfg, checkpoint):
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_id']); tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(cfg['model_id'],torch_dtype=torch.float16).cuda().eval(); base.requires_grad_(False)
    install_lora(base,cfg['lora_rank'],cfg['lora_alpha'],cfg['lora_dropout']); connector=Connector(cfg['feature_dim'],cfg['hidden_dim']).cuda().eval()
    saved=torch.load(Path(checkpoint)/'trainer.pt',map_location='cpu',weights_only=False); connector.load_state_dict(saved['connector'])
    for name,p in base.named_parameters():
        if name in saved['model']: p.data.copy_(saved['model'][name].to(p))
    return VLM(base,connector).eval(),tokenizer

@torch.inference_mode()
def generate(vlm, tokenizer, question, feature=None):
    question=_cap_tokens(tokenizer,question,80)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    ids=torch.tensor(prompt,device='cuda')[None]
    embeds=vlm.language_model.get_input_embeddings()(ids); mask=torch.ones(ids.shape,device='cuda',dtype=torch.long)
    if feature is not None:
        visual=vlm.connector(feature[None].cuda()).to(embeds.dtype); at=_find_subsequence(prompt,tokenizer(question,add_special_tokens=False).input_ids)
        embeds=torch.cat((embeds[:,:at],visual,embeds[:,at:]),1); mask=torch.ones(embeds.shape[:2],device='cuda',dtype=torch.long)
    output=vlm.language_model.generate(inputs_embeds=embeds,attention_mask=mask,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0],skip_special_tokens=True).strip()

@torch.inference_mode()
def reference_nll(vlm, tokenizer, row, feature=None):
    """Per-example answer NLL; visual/prompt/termination masking matches training."""
    answer=row.get('answer',row.get('caption','')); limit=64 if row.get('kind')=='caption' else 16
    tokens,prompt,visual_position=render(tokenizer,row.get('question','Describe this image.'),answer,limit)
    ids=tokens[None].cuda(); labels=torch.full_like(ids,-100); labels[:,prompt:]=ids[:,prompt:]
    kwargs={'input_ids':ids,'attention_mask':torch.ones_like(ids),'position_ids':torch.arange(ids.shape[1],device='cuda')[None], 'labels':labels}
    if feature is not None: kwargs.update(visual_features=feature[None].cuda(),visual_positions=[visual_position])
    return float(vlm(**kwargs).loss)


def record_key(record):
    return (record['pair_id'], int(record['image_id']), record['condition'])


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fixed_mismatch_mapping(rows, path):
    """Persist one unrelated-image control per complete pair before generation."""
    pair_images = {}
    for row in rows:
        pair_images.setdefault(row['pair_id'], set()).update((int(row['image_id']), int(row['partner_image_id'])))
    pair_ids = sorted(pair_images)
    if len(pair_ids) < 2:
        raise ValueError('A mismatched-image control requires at least two pairs.')
    if path.exists():
        mapping = {pair_id: int(image_id) for pair_id, image_id in read_json(path).items()}
        if set(mapping) != set(pair_ids):
            raise ValueError(f'Existing mismatch mapping at {path} belongs to another evaluation split.')
    else:
        mapping = {}
        for index, pair_id in enumerate(pair_ids):
            other_pair = pair_ids[(index + 1) % len(pair_ids)]
            mapping[pair_id] = min(pair_images[other_pair])
        write_json(path, mapping)
    for pair_id, image_id in mapping.items():
        if image_id in pair_images[pair_id]:
            raise ValueError('Mismatched-image mapping selected an image from the original pair.')
    return mapping


def append_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def summarize_suite(records):
    grouped = {condition: [record for record in records if record['condition'] == condition] for condition in CONDITIONS}
    pair_ids = sorted({record['pair_id'] for record in records})
    by_pair = {
        pair_id: {condition: [record for record in grouped[condition] if record['pair_id'] == pair_id]
                  for condition in CONDITIONS}
        for pair_id in pair_ids
    }
    if any(len(entries) != 2 for values in by_pair.values() for entries in values.values()):
        raise RuntimeError('Cannot summarize a partial evaluation; resume until both members of every pair are recorded.')
    correct = [sum(record['score'] for record in by_pair[pair_id]['correct']) / 2 for pair_id in pair_ids]
    swap = [sum(record['score'] for record in by_pair[pair_id]['swap']) / 2 for pair_id in pair_ids]
    pair_accuracy = {condition: sum(all(record['score'] > 0 for record in by_pair[pair_id][condition]) for pair_id in pair_ids) / len(pair_ids) for condition in CONDITIONS}
    identical = {condition: sum(entries[0]['prediction'] == entries[1]['prediction'] for entries in (by_pair[pair_id][condition] for pair_id in pair_ids)) / len(pair_ids) for condition in CONDITIONS}
    summary = {
        'correct_vs_swap_accuracy': bootstrap_pair_difference(correct, swap),
        'mean_accuracy': {condition: sum(record['score'] for record in grouped[condition]) / len(grouped[condition]) for condition in CONDITIONS},
        'pair_accuracy': pair_accuracy,
        'identical_answer_rate': identical,
        'empty_answer_rate': {condition: sum(not record['prediction'] for record in grouped[condition]) / len(grouped[condition]) for condition in CONDITIONS},
        'mean_reference_nll': {condition: sum(record['reference_nll'] for record in grouped[condition]) / len(grouped[condition]) for condition in CONDITIONS},
        'mean_seconds_per_answer': {condition: sum(record['seconds'] for record in grouped[condition]) / len(grouped[condition]) for condition in CONDITIONS},
    }
    summary['swap_penalty_nll'] = summary['mean_reference_nll']['swap'] - summary['mean_reference_nll']['correct']
    return summary


def evaluate_suite(vlm, tokenizer, rows, suite, out, cfg):
    examples_path = out / f'{suite}_examples.jsonl'
    mapping = fixed_mismatch_mapping(rows, out / f'{suite}_mismatch_mapping.json')
    records = load_jsonl(examples_path)
    completed = {record_key(record) for record in records}
    expected = len(rows) * len(CONDITIONS)
    if len(completed) != len(records):
        raise ValueError(f'Duplicate records in {examples_path}; remove that file before retrying.')
    if records:
        print(f'Resuming {suite} evaluation: {len(records)}/{expected} condition records already saved.', flush=True)
    for index, row in enumerate(rows):
        images = {'correct': row['image_id'], 'swap': row['partner_image_id'], 'mismatch': mapping[row['pair_id']], 'no_image': None}
        for condition, image_id in images.items():
            key = (row['pair_id'], int(row['image_id']), condition)
            if key in completed:
                continue
            started = time.perf_counter()
            feature = None if image_id is None else torch.load(Path(cfg['feature_dir']) / f'{image_id}.pt', map_location='cpu', weights_only=True)
            prediction = generate(vlm, tokenizer, row.get('question', ''), feature)
            score = vqa_soft_score(prediction, row['answers']) if suite == 'natural' else exact_score(prediction, row['answer'])
            nll = reference_nll(vlm, tokenizer, row, feature)
            record = {'suite': suite, 'condition': condition, 'pair_id': row['pair_id'], 'image_id': row['image_id'], 'shown_image_id': image_id, 'question': row.get('question', ''), 'target': row['answer'], 'prediction': prediction, 'score': score, 'reference_nll': nll, 'seconds': time.perf_counter() - started, 'kind': row.get('kind', row.get('answer_type'))}
            append_record(examples_path, record)
            records.append(record)
            completed.add(key)
        print(f'{suite}: {index + 1}/{len(rows)} examples ({len(records)}/{expected} condition records)', flush=True)
    write_json(out / f'{suite}_summary.json', summarize_suite(records))
    print(f'Saved {suite} final-evaluation summary.', flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/task2/vlm_196.json'); p.add_argument('--checkpoint',required=True); p.add_argument('--split',choices=('dev','final'),default='final'); args=p.parse_args(); cfg=read_json(args.config)
    manifest=read_manifest(Path(cfg['prepared_dir'])/'manifest.json'); vlm,tokenizer=load_vlm(cfg,args.checkpoint); out=Path(cfg['results_dir'])/f'{args.split}_evaluation'; out.mkdir(parents=True,exist_ok=True)
    for suite,key in (('annotation',f'annotation_{args.split}'),('natural',f'natural_{args.split}')):
        evaluate_suite(vlm, tokenizer, manifest[key], suite, out, cfg)
if __name__=='__main__': main()
