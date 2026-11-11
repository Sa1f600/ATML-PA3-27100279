import pytest
from task2_vlm.metrics import bootstrap_pair_difference, exact_score, normalize_answer, vqa_soft_score
from task2_vlm.model import Connector
from task2_vlm.train import CurriculumSampler, gradients_are_finite
from task2_vlm.data import annotation_pairs, complementary_pairs, consensus
from task2_vlm.evaluate import fixed_mismatch_mapping, summarize_suite

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


def test_vqa_pairing_is_linear_and_uses_an_original_consensus_answer():
    def annotation(question_id, image_id, answer):
        return {"question_id": question_id, "image_id": image_id, "answer_type": "yes/no", "answers": [{"answer": answer}] * 8 + [{"answer": "noise"}] * 2}
    questions = {"questions": [
        {"question_id": 1, "image_id": 101, "question": "Is this red?"},
        {"question_id": 2, "image_id": 102, "question": "Is this red?"},
        {"question_id": 3, "image_id": 103, "question": "Is this red?"},
        {"question_id": 4, "image_id": 104, "question": "Is this red?"},
    ]}
    annotations = {"annotations": [annotation(1, 101, "Yes"), annotation(2, 102, "No"), annotation(3, 103, "Yes"), annotation(4, 104, "No")]}
    target, agreement, _ = consensus(annotations["annotations"][0])
    assert target == "Yes" and agreement == 8
    rows = complementary_pairs(questions, annotations, {"yesno": 2, "number": 0, "other": 0}, set(), 6304, label="test VQAv2")
    assert len(rows) == 4
    assert {row["image_id"] for row in rows} == {101, 102, 103, 104}


def test_annotation_pair_builder_uses_indexes_and_keeps_eval_images_disjoint():
    annotations = []
    # Four distinct count buckets for category 1, plus enough images that do
    # not contain it for a presence pair.
    for image_id, count in enumerate((1, 2, 3, 4), start=1):
        annotations.extend({"image_id": image_id, "category_id": 1} for _ in range(count))
    instances = {"categories": [{"id": 1, "name": "cat"}], "annotations": annotations}
    rows = annotation_pairs(instances, list(range(1, 9)), 1, set(), 6304, label="test COCO")
    assert len(rows) == 4  # one presence pair and one counting pair
    assert len({row["image_id"] for row in rows}) == 4
    assert {row["kind"] for row in rows} == {"presence", "counting"}


def test_gradient_finiteness_guard_detects_overflow_before_a_step():
    import torch
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([1.0])
    assert gradients_are_finite([parameter])
    parameter.grad = torch.tensor([float("inf")])
    assert not gradients_are_finite([parameter])


def test_final_evaluation_mismatch_mapping_is_fixed_and_excludes_each_pair(tmp_path):
    rows = [
        {"pair_id": "p1", "image_id": 1, "partner_image_id": 2},
        {"pair_id": "p1", "image_id": 2, "partner_image_id": 1},
        {"pair_id": "p2", "image_id": 3, "partner_image_id": 4},
        {"pair_id": "p2", "image_id": 4, "partner_image_id": 3},
    ]
    path = tmp_path / "mapping.json"
    mapping = fixed_mismatch_mapping(rows, path)
    assert mapping == fixed_mismatch_mapping(rows, path)
    assert mapping["p1"] not in {1, 2}
    assert mapping["p2"] not in {3, 4}


def test_final_evaluation_summary_requires_complete_pairs():
    records = []
    for pair_id in ("p1", "p2"):
        for condition in ("correct", "swap", "mismatch", "no_image"):
            for image_id in (1, 2):
                records.append({"pair_id": pair_id, "image_id": image_id, "condition": condition,
                                "prediction": "yes", "score": int(condition == "correct"),
                                "reference_nll": 1.0, "seconds": 0.1})
    summary = summarize_suite(records)
    assert summary["mean_accuracy"]["correct"] == 1
    assert summary["mean_accuracy"]["swap"] == 0
