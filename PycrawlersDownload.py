#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from pycrawlers import huggingface

if __name__ == '__main__':


    my_token = 'get your own token from explorer'
    hf = huggingface(token=my_token)

    urls = [
            'https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf', 
            'https://huggingface.co/datasets/HPAI-BSC/CareQA', 
            'https://huggingface.co/datasets/openlifescienceai/medmcqa',
            'https://huggingface.co/datasets/qiaojin/PubMedQA', 
            'https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA',
            'https://huggingface.co/datasets/LangAGI-Lab/medbullets', # 4 options
            'https://huggingface.co/datasets/jdh-algo/JMED', 
            'https://huggingface.co/datasets/bluesky333/MedExQA', 
            ]
    # construct the paths
    dataset_root = 'opencompass/adataset/meddata'
    paths = [os.path.join(dataset_root, url.split('/')[-1]) for url in urls]
    urls = [os.path.join(url, 'tree/main') for url in urls]

    hf.get_batch_data(urls, paths)


    print("GG, restarting")
