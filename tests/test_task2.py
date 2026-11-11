import pytest
from task2_vlm.metrics import bootstrap_pair_difference, exact_score, normalize_answer, vqa_soft_score
from task2_vlm.model import Connector

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
