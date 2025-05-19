from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedExQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

medexqa_all_sets = [
    'biomedical_engineer',
    'clinical_laboratory_scientist',
    'clinical_psychologist',
    'occupational_therapist',
    'speech_pathologist',
]

medexqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev', 
    test_split='test',)

medexqa_datasets = []
for _name in medexqa_all_sets:
    medexqa_infer_cfg = dict(
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

    medexqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

    medexqa_datasets.append(
        dict(
            abbr=f'medexqa_{_name}',
            type=MedExQADataset,
            path='/root/path/to/datasets/meddata/bluesky333/MedExQA',
            name=_name,
            reader_cfg=medexqa_reader_cfg,
            infer_cfg=medexqa_infer_cfg,
            eval_cfg=medexqa_eval_cfg,
        ))
del _name

datasets = [*medexqa_datasets]