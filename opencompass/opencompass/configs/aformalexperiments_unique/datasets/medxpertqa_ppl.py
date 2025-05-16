from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedXpertQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

medxpertqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    output_column='target',
    train_split='dev',
    test_split='test',
    )

medxpertqa_datasets = []

medxpertqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': 'The following makes sense: \nQ: {input}\nA: {A}\n',
            'B': 'The following makes sense: \nQ: {input}\nA: {B}\n',
            'C': 'The following makes sense: \nQ: {input}\nA: {C}\n',
            'D': 'The following makes sense: \nQ: {input}\nA: {D}\n',
            'E': 'The following makes sense: \nQ: {input}\nA: {E}\n',
            'F': 'The following makes sense: \nQ: {input}\nA: {F}\n',
            'G': 'The following makes sense: \nQ: {input}\nA: {G}\n',
            'H': 'The following makes sense: \nQ: {input}\nA: {H}\n',
            'I': 'The following makes sense: \nQ: {input}\nA: {I}\n',
            'J': 'The following makes sense: \nQ: {input}\nA: {J}\n',
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

medxpertqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

medxpertqa_datasets.append(
    dict(
        abbr=f'medxpertqa',
        type=MedXpertQADataset,
        path='/home/gsb/opencompass/adatasets/meddata/TsinghuaC3I/MedXpertQA/Text',
        name='medxpertqa',
        reader_cfg=medxpertqa_reader_cfg,
        infer_cfg=medxpertqa_infer_cfg,
        eval_cfg=medxpertqa_eval_cfg,
    ))


datasets = [*medxpertqa_datasets]