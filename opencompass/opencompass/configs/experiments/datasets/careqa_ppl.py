from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import CareQADataset

careqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev',
    test_split='test',
    )

careqa_datasets = []

careqa_infer_cfg = dict(

    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': 'The following makes sense: \nQ: {input}\nA: {A}\n',
            'B': 'The following makes sense: \nQ: {input}\nA: {B}\n',
            'C': 'The following makes sense: \nQ: {input}\nA: {C}\n',
            'D': 'The following makes sense: \nQ: {input}\nA: {D}\n',
        }
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

careqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

careqa_datasets.append(
    dict(
        abbr=f'careqa',
        type=CareQADataset,
        path='/root/path/to/datasets/meddata/CareQA',
        name='CareQA_en',
        reader_cfg=careqa_reader_cfg,
        infer_cfg=careqa_infer_cfg,
        eval_cfg=careqa_eval_cfg,
    ))


datasets = [*careqa_datasets]