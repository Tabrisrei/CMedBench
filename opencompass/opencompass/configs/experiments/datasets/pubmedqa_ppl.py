from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import PubMedQADataset                                
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

pubmedqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C'],
    output_column='target',
    # train_split='train',
    # test_split='validation',
    )

pubmedqa_datasets = []

pubmedqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': 'The following makes sense: \nQ: {input}\nA: {A}\n',
            'B': 'The following makes sense: \nQ: {input}\nA: {B}\n',
            'C': 'The following makes sense: \nQ: {input}\nA: {C}\n',
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

pubmedqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

pubmedqa_datasets.append(
    dict(
        abbr=f'pubmedqa',
        type=PubMedQADataset,
        path='/root/path/to/datasets/meddata/qiaojin/PubMedQA/pqa_labeled',
        name='pubmedqa',
        reader_cfg=pubmedqa_reader_cfg,
        infer_cfg=pubmedqa_infer_cfg,
        eval_cfg=pubmedqa_eval_cfg,
    ))


datasets = [*pubmedqa_datasets]
