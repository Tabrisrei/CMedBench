from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

medqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev',
    test_split='test',)

medqa_datasets = []

medqa_infer_cfg = dict(
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

medqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

medqa_datasets.append(
    dict(
        abbr=f'medqa',
        type=MedQADataset,
        path='/home/gsb/opencompass/adatasets/meddata/GBaker/MedQA-USMLE-4-options-hf',
        name='MedQA-USMLE-4-options-hf',
        reader_cfg=medqa_reader_cfg,
        infer_cfg=medqa_infer_cfg,
        eval_cfg=medqa_eval_cfg,
    ))


datasets = [*medqa_datasets]