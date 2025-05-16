from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedbulletsOp5Dataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

medbulletsop5_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D', 'E'],
    output_column='target',
    # train_split='dev',
    # test_split='test',
    )

medbulletsop5_datasets = []

medbulletsop5_infer_cfg = dict(

    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': 'The following makes sense: \nQ: {input}\nA: {A}\n',
            'B': 'The following makes sense: \nQ: {input}\nA: {B}\n',
            'C': 'The following makes sense: \nQ: {input}\nA: {C}\n',
            'D': 'The following makes sense: \nQ: {input}\nA: {D}\n',
            'E': 'The following makes sense: \nQ: {input}\nA: {E}\n',
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

medbulletsop5_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

medbulletsop5_datasets.append(
    dict(
        abbr=f'medbulletsop5',
        type=MedbulletsOp5Dataset,
        path='/home/gsb/opencompass/adatasets/meddata/LangAGI-Lab/medbullets_op5/data',
        name='medbulletsop5',
        reader_cfg=medbulletsop5_reader_cfg,
        infer_cfg=medbulletsop5_infer_cfg,
        eval_cfg=medbulletsop5_eval_cfg,
    ))

datasets = [*medbulletsop5_datasets]