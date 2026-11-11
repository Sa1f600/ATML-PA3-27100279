import pytest
from task2_vlm.metrics import bootstrap_pair_difference, exact_score, normalize_answer, vqa_soft_score
from task2_vlm.model import Connector
from task2_vlm.train import CurriculumSampler

def test_vqa_normalization_and_soft_consensus():
    assert normalize_answer('The, Two!') == '2'
    assert exact_score('two', '2') == 1
    assert vqa_soft_score('cat', ['cat']*10) == 1
    assert vqa_soft_score('dog', ['cat']*10) == 0

def test_complete_pair_bootstrap_requires_aligned_pairs():
    result=bootstrap_pair_difference([1,0,1],[0,0,0],resamples=32)
    assert result['difference']==pytest.approx(2/3) and len(result['ci95'])==2
    with pytest.raises(ValueError): bootstrap_pair_difference([1],[1,0])

def test_connector_preserves_visual_sequence_shape():
    import torch
    connector=Connector(); out=connector(torch.zeros(2,196,768))
    assert out.shape==(2,196,1024)

def test_connector_scale_is_applied():
    import torch
    connector=Connector(); connector.output_scale.fill_(2)
    x=torch.zeros(1,196,768)
    assert torch.allclose(connector(x),2*connector.layers(x))

def test_curriculum_sampler_restores_the_exact_next_examples():
    captions=[{'image_id':i,'kind':'caption','caption':str(i)} for i in range(8)]
    pairs=[{'image_id':10,'kind':'presence','pair_id':'p','question':'q','answer':'yes'}, {'image_id':11,'kind':'presence','pair_id':'p','question':'q','answer':'no'}, {'image_id':12,'kind':'counting','pair_id':'c','question':'q','answer':'1'}, {'image_id':13,'kind':'counting','pair_id':'c','question':'q','answer':'2'}]
    manifest={'caption_train':captions,'annotation_train':pairs,'natural_train':pairs}
    first=CurriculumSampler(manifest,'grounding',6304)
    first.next_examples(8); saved=first.state_dict(); expected=first.next_examples(8)
    resumed=CurriculumSampler(manifest,'grounding',6304); resumed.load_state_dict(saved)
    assert resumed.next_examples(8)==expected
