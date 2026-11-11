"""Three-stage curriculum training with resumable connector/LoRA checkpoints."""
import argparse, json, random, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.experiment import capture_rng, read_json, restore_rng, seed_everything, write_json
from task2_vlm.data import read_json as read_manifest
from task2_vlm.model import Connector, VLM, install_lora

STAGES = ('caption', 'grounding', 'natural')

def _cap_tokens(tokenizer, text, limit):
    ids=tokenizer(str(text),add_special_tokens=False).input_ids[:limit]
    return tokenizer.decode(ids,skip_special_tokens=True)

def _find_subsequence(haystack, needle):
    for index in range(len(haystack)-len(needle)+1):
        if haystack[index:index+len(needle)] == needle: return index
    raise RuntimeError('Could not locate the user question in Qwen chat tokens.')

def render(tokenizer, question, answer, max_target):
    question = _cap_tokens(tokenizer, question, 80); answer = _cap_tokens(tokenizer, answer, max_target)
    prompt = tokenizer.apply_chat_template([{'role':'user','content':question}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template([{'role':'user','content':question},{'role':'assistant','content':answer}], tokenize=True, add_generation_prompt=False, enable_thinking=False)
    question_ids=tokenizer(question,add_special_tokens=False).input_ids
    return torch.tensor(full), len(prompt), _find_subsequence(prompt,question_ids)

def collate(rows, tokenizer, feature_dir, device):
    encoded = [render(tokenizer, r.get('question','Describe this image.'), r.get('answer', r.get('caption','')), 64 if r['kind']=='caption' else 16) for r in rows]
    length = max(len(ids) for ids,_,_ in encoded); ids = torch.full((len(rows),length), tokenizer.pad_token_id, dtype=torch.long, device=device); labels = torch.full_like(ids,-100); mask=torch.zeros_like(ids)
    positions=[]
    for i,(tokens,prompt,visual_position) in enumerate(encoded):
        ids[i,:len(tokens)]=tokens.to(device); mask[i,:len(tokens)]=1; labels[i,prompt:len(tokens)]=tokens[prompt:]; positions.append(visual_position)
    features=torch.stack([torch.load(Path(feature_dir)/f"{r['image_id']}.pt", map_location='cpu', weights_only=True) for r in rows]).to(device)
    return dict(input_ids=ids, attention_mask=mask, position_ids=torch.arange(length,device=device).expand(len(rows),-1), labels=labels, visual_features=features, visual_positions=positions)

def stage_rows(manifest, stage, rng):
    if stage=='caption':
        rows=list(manifest['caption_train']); rng.shuffle(rows)
        for row in rows*2: yield row
    annotations = manifest['annotation_train']; presence = [r for r in annotations if r['kind']=='presence']; counting = [r for r in annotations if r['kind']=='counting']
    pools = [(presence,.50),(counting,.25),(manifest['caption_train'],.25)] if stage=='grounding' else [(manifest['natural_train'],.70),(presence,.10),(counting,.10),(manifest['caption_train'],.10)]
    while True:
        draw=rng.random(); cumulative=0
        for rows,prob in pools:
            cumulative+=prob
            if draw<=cumulative:
                pair=rng.choice(rows).get('pair_id')
                if pair: yield from [r for r in rows if r.get('pair_id')==pair]
                else: yield random.choice(rows); yield random.choice(rows)
                break

def checkpoint(path, vlm, optimizer, scaler, state, sampler_rng, cfg):
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_name(path.name+'.saving'); tmp.mkdir(parents=True,exist_ok=False)
    torch.save({'connector':vlm.connector.state_dict(),'model':{n:p.detach().cpu() for n,p in vlm.language_model.named_parameters() if '.a.weight' in n or '.b.weight' in n},'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'state':state,'rng':capture_rng(),'sampler_rng':sampler_rng.getstate()},tmp/'trainer.pt')
    write_json(tmp/'config.json',cfg); tmp.rename(path)

@torch.inference_mode()
def calibrate_connector(connector, base, manifest, cfg):
    rows=manifest['caption_train'][:128]
    features=torch.stack([torch.load(Path(cfg['feature_dir'])/f"{row['image_id']}.pt",map_location='cpu',weights_only=True) for row in rows]).cuda()
    raw=connector.layers(features.float()).square().mean().sqrt().clamp_min(1e-8)
    text=base.get_input_embeddings()(torch.tensor([0],device='cuda')).float().square().mean().sqrt()
    connector.output_scale.copy_((text/raw).detach().cpu())
    return {'rule':'RMS-match raw connector outputs to frozen text embeddings on first 128 selected training captions','text_embedding_rms':text.item(),'connector_raw_rms':raw.item(),'output_scale':connector.output_scale.item()}

@torch.inference_mode()
def development_metrics(vlm, tokenizer, manifest, cfg):
    def nll(row, image_id):
        sample=dict(row); sample['image_id']=image_id
        batch=collate([sample],tokenizer,cfg['feature_dir'],'cuda')
        if image_id is None: batch['visual_features']=None; batch['visual_positions']=None
        return float(vlm(**batch).loss)
    result={'caption_reference_loss':sum(nll(row,row['image_id']) for row in manifest['caption_dev'])/len(manifest['caption_dev'])}
    for suite,key in (('annotation','annotation_dev'),('natural','natural_dev')):
        correct=[nll(row,row['image_id']) for row in manifest[key]]; swap=[nll(row,row['partner_image_id']) for row in manifest[key]]; no_image=[nll(row,None) for row in manifest[key]]
        result[f'{suite}_answer_nll']={'correct':sum(correct)/len(correct),'swap':sum(swap)/len(swap),'no_image':sum(no_image)/len(no_image),'swap_penalty':sum(b-a for a,b in zip(correct,swap))/len(correct)}
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/task2/vlm_196.json'); p.add_argument('--resume',type=Path); args=p.parse_args(); cfg=read_json(args.config)
    if not torch.cuda.is_available(): raise RuntimeError('Task 2 requires CUDA FP16.')
    seed_everything(cfg['seed']); device='cuda'; manifest=read_manifest(Path(cfg['prepared_dir'])/'manifest.json')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_id']); tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(cfg['model_id'],torch_dtype=torch.float16).to(device); base.gradient_checkpointing_enable(); base.requires_grad_(False)
    connector=Connector(cfg['feature_dim'],cfg['hidden_dim']).to(device); vlm=VLM(base,connector); targets=[]
    state={'stage_index':0,'stage_update':0,'global_update':0,'examples_seen':0}; scaler=torch.amp.GradScaler('cuda'); rng=random.Random(cfg['seed'])
    saved=None
    if args.resume:
        saved=torch.load(args.resume/'trainer.pt',map_location='cpu',weights_only=False); state=saved['state']
        if state['stage_index'] >= 1: targets=install_lora(base,cfg['lora_rank'],cfg['lora_alpha'],cfg['lora_dropout'])
        connector.load_state_dict(saved['connector'])
        for name, parameter in base.named_parameters():
            if name in saved['model']: parameter.data.copy_(saved['model'][name].to(parameter))
        restore_rng(saved['rng']); rng.setstate(saved['sampler_rng'])
    results=Path(cfg['results_dir']); results.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if saved is None: write_json(results/'connector_calibration.json',calibrate_connector(connector,base,manifest,cfg))
    def log_development(tag):
        record={'tag':tag,'global_update':state['global_update'],**development_metrics(vlm,tokenizer,manifest,cfg)}
        with (results/'development.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
    if saved is None: log_development('before_training')
    for stage_index in range(state['stage_index'],len(STAGES)):
        stage=STAGES[stage_index]; spec=cfg['stages'][stage]
        if stage!='caption' and not targets: targets=install_lora(base,cfg['lora_rank'],cfg['lora_alpha'],cfg['lora_dropout'])
        groups=[{'params':list(connector.parameters()),'lr':spec['connector_lr']}]
        if stage!='caption': groups.append({'params':[p for p in base.parameters() if p.requires_grad],'lr':spec['lora_lr']})
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
        params=[parameter for group in groups for parameter in group['params']]
        if saved is not None and stage_index==state['stage_index']: optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        iterator=stage_rows(manifest,stage,rng); start=state['stage_update'] if stage_index==state['stage_index'] else 0
        for update in range(start,spec['updates']):
            # Per-stage 5% linear warmup, then cosine decay to 10% of the peak.
            warmup=max(1,round(.05*spec['updates'])); progress=(update+1)/spec['updates']
            scale=(update+1)/warmup if update<warmup else .1+.9*.5*(1+__import__('math').cos(__import__('math').pi*(update+1-warmup)/(spec['updates']-warmup)))
            for group, peak in zip(optimizer.param_groups,[spec['connector_lr'],spec['lora_lr']]): group['lr']=peak*scale
            optimizer.zero_grad(set_to_none=True); losses=[]
            for _ in range(cfg['gradient_accumulation']):
                rows=[next(iterator) for _ in range(cfg['microbatch_size'])]; batch=collate(rows,tokenizer,cfg['feature_dir'],device)
                with torch.autocast('cuda',torch.float16): loss=vlm(**batch).loss/cfg['gradient_accumulation']
                scaler.scale(loss).backward(); losses.append(loss.item()*cfg['gradient_accumulation']); state['examples_seen']+=len(rows)
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip_norm']); scaler.step(optimizer); scaler.update(); state.update(stage_index=stage_index,stage_update=update+1,global_update=state['global_update']+1)
            with (results/'training.jsonl').open('a') as f: f.write(json.dumps({'stage':stage,'global_update':state['global_update'],'stage_update':update+1,'loss':sum(losses)/len(losses),'connector_lr':optimizer.param_groups[0]['lr'],'lora_lr':optimizer.param_groups[-1]['lr'] if stage!='caption' else 0,'examples_seen':state['examples_seen'],'time':time.time()})+'\n')
            endpoint=update+1==spec['updates']
            if state['global_update']%250==0 or endpoint:
                path=Path(cfg['checkpoint_dir'])/f"update_{state['global_update']:04d}"; checkpoint(path,vlm,optimizer,scaler,state,rng,cfg)
                if endpoint:
                    torch.save(vlm.connector.state_dict(),path/'connector.pt')
                    torch.save({name:p.detach().cpu() for name,p in base.named_parameters() if p.requires_grad},path/'adapter.pt')
            if state['global_update']%500==0 or endpoint:
                log_development(f"update_{state['global_update']:04d}")
                write_json(results/'resources.json',{'global_update':state['global_update'],'stage':stage,'training_seconds':time.perf_counter()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'trainable_parameters':sum(p.numel() for p in params)})
        state['stage_index']=stage_index+1; state['stage_update']=0
    print('Task 2 curriculum complete.')
if __name__=='__main__': main()
