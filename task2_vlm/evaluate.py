"""Final four-image-condition grounding evaluation for Task 2."""
import argparse, json, time
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

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/task2/vlm_196.json'); p.add_argument('--checkpoint',required=True); p.add_argument('--split',choices=('dev','final'),default='final'); args=p.parse_args(); cfg=read_json(args.config)
    manifest=read_manifest(Path(cfg['prepared_dir'])/'manifest.json'); vlm,tokenizer=load_vlm(cfg,args.checkpoint); out=Path(cfg['results_dir'])/f'{args.split}_evaluation'; out.mkdir(parents=True,exist_ok=True)
    for suite,key in (('annotation',f'annotation_{args.split}'),('natural',f'natural_{args.split}')):
        rows=manifest[key]; available=sorted({r['image_id'] for r in rows}); records=[]
        for index,row in enumerate(rows):
            mismatch=next(image for image in available if image not in (row['image_id'],row['partner_image_id']))
            images={'correct':row['image_id'],'swap':row['partner_image_id'],'mismatch':mismatch,'no_image':None}
            for condition,image_id in images.items():
                started=time.perf_counter(); feature=None if image_id is None else torch.load(Path(cfg['feature_dir'])/f'{image_id}.pt',map_location='cpu',weights_only=True)
                prediction=generate(vlm,tokenizer,row.get('question',''),feature); score=vqa_soft_score(prediction,row['answers']) if suite=='natural' else exact_score(prediction,row['answer'])
                nll=reference_nll(vlm,tokenizer,row,feature)
                records.append({'suite':suite,'condition':condition,'pair_id':row['pair_id'],'image_id':row['image_id'],'shown_image_id':image_id,'question':row.get('question',''),'target':row['answer'],'prediction':prediction,'score':score,'reference_nll':nll,'seconds':time.perf_counter()-started,'kind':row.get('kind',row.get('answer_type'))})
            print(f'{suite}: {index+1}/{len(rows)}',flush=True)
        with (out/f'{suite}_examples.jsonl').open('w') as stream:
            for record in records: stream.write(json.dumps(record)+'\n')
        pair_scores = {}
        for pair_id in sorted({record['pair_id'] for record in records}):
            pair_scores[pair_id] = {condition: [record['score'] for record in records
                                                 if record['pair_id'] == pair_id and record['condition'] == condition]
                                    for condition in CONDITIONS}
        correct = [sum(scores['correct']) / len(scores['correct']) for scores in pair_scores.values()]
        swap = [sum(scores['swap']) / len(scores['swap']) for scores in pair_scores.values()]
        grouped={c:[r for r in records if r['condition']==c] for c in CONDITIONS}
        pair_accuracy={c:sum(all(r['score'] > 0 for r in grouped[c] if r['pair_id']==pair) for pair in pair_scores)/len(pair_scores) for c in CONDITIONS}
        identical={c:sum(grouped[c][i]['prediction']==grouped[c][i+1]['prediction'] for i in range(0,len(grouped[c]),2))/len(pair_scores) for c in CONDITIONS}
        empty={c:sum(not r['prediction'] for r in grouped[c])/len(grouped[c]) for c in CONDITIONS}
        summary={'correct_vs_swap_accuracy':bootstrap_pair_difference(correct,swap),'mean_accuracy':{c:sum(r['score'] for r in grouped[c])/len(grouped[c]) for c in CONDITIONS},'pair_accuracy':pair_accuracy,'identical_answer_rate':identical,'empty_answer_rate':empty,'mean_reference_nll':{c:sum(r['reference_nll'] for r in grouped[c])/len(grouped[c]) for c in CONDITIONS},'mean_seconds_per_answer':{c:sum(r['seconds'] for r in grouped[c])/len(grouped[c]) for c in CONDITIONS}}
        summary['swap_penalty_nll']=summary['mean_reference_nll']['swap']-summary['mean_reference_nll']['correct']
        write_json(out/f'{suite}_summary.json',summary)
if __name__=='__main__': main()
