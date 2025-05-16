from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedMCQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

medmcqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='train',
    test_split='validation',
    )

medmcqa_datasets = []

medmcqa_infer_cfg = dict(

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

medmcqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

medmcqa_datasets.append(
    dict(
        abbr=f'medmcqa',
        type=MedMCQADataset,
        path='/home/gsb/opencompass/adatasets/meddata/openlifescienceai/medmcqa/data',
        name='medmcqa',
        reader_cfg=medmcqa_reader_cfg,
        infer_cfg=medmcqa_infer_cfg,
        eval_cfg=medmcqa_eval_cfg,
    ))


datasets = [*medmcqa_datasets]