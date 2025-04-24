# GeoGen

A pipeline for the automatic construction of geometry problems along with step-by-step solutions.


## Resources

Paper: [Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration](https://arxiv.org/pdf/2504.12773)  
Dataset: [GeoExpand & GeoSynth](https://huggingface.co/datasets/ycpNotFound/GeoGen)  

[![arXiv](https://img.shields.io/badge/arXiv-2404.12345-B31B1B.svg)](https://arxiv.org/pdf/2504.12773)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/ycpNotFound/GeoGen)


## Features

- GeoGen can automatically synthesize geometry diagrams.
- GeoGen can generate questions and step-by-step solutions by incorporating symbolic systems, applied to both public datasets (Geometry3K, PGPS9K) and our synthetic data.
- GeoGen is inspired by [AlphaGeometry](https://github.com/google-deepmind/alphageometry), and leverages [FormalGeo](https://github.com/FormalGeo/FormalGeo) as the underlying symbolic system.

![Framework of our GeoGen pipeline.](GeoGen.jpg)

> As the symbolic system relies on manually defined theorem rules, and symbolic annotations may be missing in the dataset, some types of geometry problems may remain unsolvable. We welcome issues and feedback to help us identify and address such limitations.

## Getting Started

```bash
conda create -n geogen python=3.9
conda activate geogen
git clone https://github.com/ycpNotFound/GeoGen.git
cd GeoGen
pip install -r requirements.txt
```

## Core components
```bash
GeoGen/
│
├── generator.py        # Randomly sample premise and generate literals
├── allocator.py        # Assign coordinates for literals
├── plotter.py          # Draw with geometry diagram
├── solver.py           # symbolic solver with our improved forward-search
├── target_finder.py    # Conduct reasoning, find target and create question & answer
└── ...
```

## Datasets

We collect symbolic annotations for most of the diagrams from Geometry3K and PGPS9K, as listed in `datasets_info`. Note that symbolic reasoning can be conducted without relying on images. For Geometry3K, we utilize annotations from `formalgeo7k`, which provides detailed symbolic annotations for public datasets from [FormalGeo](https://github.com/FormalGeo/FormalGeo). For PGPS9K, we use regularization tools and prompt LLMs to convert the source annotations into the required symbolic format. Since the annotations are constructed automatically, some omissions may occur. However, these do not affect the overall workflow.

You can download images and original annotations for Geometry3K from [Inter-GPS](https://github.com/lupantech/InterGPS).

You can download images and original annotations for PGPS9K from [PGPS](https://github.com/mingliangzhang2018/PGPS).

## Create Q&A for Geometry3K & PGPS9K

Run `main_search_public.py` with multiprocessing to expand more target and create question & answer from public dataset (geometry3K and PGPS9K). 

```bash
python main_search_public.py \
    --dataset_name geo3k \ # or pgps9k 
    --save_dir /path/to/your/dir \
    --num_process 12 \
    --seed 1234 
```

## Synthesize Geometry Diagram and Create Q&A

Run `main_search_synth.py` with multiprocessing to synthesize geometry diagram and create question & answer for it, with one Q&A pair for each diagram.

```bash
python main_search_synth.py \
    --save_dir /path/to/your/dir \
    --num_process 12 \
    --seed 1234 \
    --use_default_sampling_num
    # you can disable this param and modify the sampling num in python sript.
```

## Acknowledgement

GeoGen is based on [FormalGeo](https://github.com/FormalGeo/FormalGeo), released under the MIT License. We extend its symbolic reasoning engine to support automatic diagram generation and reasoning path synthesis.

We also draw inspiration from [AlphaGeometry](https://github.com/google-deepmind/alphageometry) in designing our pipeline.

We gratefully acknowledge the use of [ms-swift](https://github.com/modelscope/ms-swift) for model training.