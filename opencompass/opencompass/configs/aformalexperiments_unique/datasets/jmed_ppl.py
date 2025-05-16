from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import JMEDDataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

jmed_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'],
    output_column='target',
    # train_split='dev',
    # test_split='test',
    )

jmed_datasets = []

jmed_infer_cfg = dict(
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
            'K': 'The following makes sense: \nQ: {input}\nA: {K}\n',
            'L': 'The following makes sense: \nQ: {input}\nA: {L}\n',
            'M': 'The following makes sense: \nQ: {input}\nA: {M}\n',
            'N': 'The following makes sense: \nQ: {input}\nA: {N}\n',
            'O': 'The following makes sense: \nQ: {input}\nA: {O}\n',
            'P': 'The following makes sense: \nQ: {input}\nA: {P}\n',
            'Q': 'The following makes sense: \nQ: {input}\nA: {Q}\n',
            'R': 'The following makes sense: \nQ: {input}\nA: {R}\n',
            'S': 'The following makes sense: \nQ: {input}\nA: {S}\n',
            'T': 'The following makes sense: \nQ: {input}\nA: {T}\n',
            'U': 'The following makes sense: \nQ: {input}\nA: {U}\n',
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

jmed_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

jmed_datasets.append(
    dict(
        abbr=f'jmed',
        type=JMEDDataset,
        path='/home/gsb/opencompass/adatasets/meddata/jdh-algo/JMED',
        name='jmed_en',
        reader_cfg=jmed_reader_cfg,
        infer_cfg=jmed_infer_cfg,
        eval_cfg=jmed_eval_cfg,
    ))

datasets = [*jmed_datasets]