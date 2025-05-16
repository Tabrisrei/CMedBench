import os
from time import time
from trustllm import config
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# from trustllm.evaluation import run_evaluation
from trustllm.task.pipeline import run_truthfulness, run_safety, run_fairness, run_robustness, run_privacy, run_ethics

trustfullness_dict = {
    'internal_consistancy': 'truthfulness/internal_codah.json',
    'external_consistancy': 'truthfulness/external.json',
    'hallucination': 'truthfulness/hallucination.json',
}

safety_dict = {
    'jailbreak': 'safety/jailbreak.json',
    'exaggerated_safety': 'safety/exaggerated_safety.json',
    'misuse': 'safety/misuse.json',
}

fairness_dict = {
    'stereotype_query_test': 'fairness/stereotype_query_test.json',
    'disparagement': 'fairness/disparagement.json',
    'preference': 'fairness/preference.json'
}

robustness_dict = {
    'advglue': 'robustness/AdvGLUE.json',
}

privacy_dict = {
    'privacy_confaide': 'privacy/privacy_awareness_confAIde.json',
    'privacy_awareness_query': 'privacy/privacy_awareness_query.json',
    'privacy_leakage': 'privacy/privacy_leakage.json',
}

ethics_dict = {
    'explicit_ethics': 'ethics/explicit_moralchoice.json',
}

if __name__ == '__main__':
    time1 = time()

    all_data_path = 'path/to/generated/result'

    truthfulness_results = run_truthfulness(  
        internal_path = \
            os.path.join(all_data_path, trustfullness_dict['internal_consistancy']), 
        external_path = \
            os.path.join(all_data_path, trustfullness_dict['external_consistancy']), 
        hallucination_path= \
            os.path.join(all_data_path, trustfullness_dict['hallucination']),  
    )

    print(f"truthfulness_results: {truthfulness_results}")

    # ==============================================================

    safety_results = run_safety(  
        jailbreak_path = \
            os.path.join(all_data_path, safety_dict['jailbreak']),  
        exaggerated_safety_path = \
            os.path.join(all_data_path, safety_dict['exaggerated_safety']),  
        misuse_path = \
            os.path.join(all_data_path, safety_dict['misuse']),  
    )  

    print(f"safety_results: {safety_results}")

    # ==============================================================

    fairness_results = run_fairness(     
        disparagement_path = \
            os.path.join(all_data_path, fairness_dict['disparagement']),     
    )  

    print(f"fairness_results: {fairness_results}")

    # ==============================================================

    robustness_results = run_robustness(  
        advglue_path = \
            os.path.join(all_data_path, robustness_dict['advglue']),
    )  

    print(f"robustness_results: {robustness_results}")

    # ==============================================================

    privacy_results = run_privacy(  
        privacy_awareness_query_path = \
            os.path.join(all_data_path, privacy_dict['privacy_awareness_query']),
        privacy_leakage_path = \
            os.path.join(all_data_path, privacy_dict['privacy_leakage']),
    )  

    print(f"privacy_results: {privacy_results}")

    # ==============================================================

    results = run_ethics(  
        explicit_ethics_path = \
            os.path.join(all_data_path, ethics_dict['explicit_ethics']),
    )  

    print(f"ethics_results: {results}")

    # ==============================================================

    time2 = time()
    print(f"Evaluation time: {time2 - time1} seconds")