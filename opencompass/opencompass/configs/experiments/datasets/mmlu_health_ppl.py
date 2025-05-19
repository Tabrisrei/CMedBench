from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

mmlu_all_sets = [
    'college_biology',
    'anatomy',
    'clinical_knowledge',
    'medical_genetics',
    'professional_medicine',
    'college_medicine',
]

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev', 
    test_split='test',)

mmlu_datasets = []
for _name in mmlu_all_sets:
    mmlu_infer_cfg = dict(
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

    mmlu_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator), )

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{_name}',
            type=MMLUDataset,
            path='/root/path/to/datasets/meddata/mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))
del _name

datasets = [*mmlu_datasets]