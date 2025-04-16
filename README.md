# GeoGen

A pipeline for the automatic construction of geometry problems along with step-by-step solutions.

## Features

- GeoGen can automatically synthesize geometry diagrams.
- GeoGen can generate questions and step-by-step solutions by incorporating symbolic systems, applied to both public datasets (Geometry3K, PGPS9K) and our synthetic data.
- GeoGen is inspired by [AlphaGeometry](https://github.com/google-deepmind/alphageometry), and leverages [FormalGeo](https://github.com/FormalGeo/FormalGeo) as the underlying symbolic system.


## Getting Started

```bash
conda create -n geogen python=3.9
git clone https://github.com/yourname/GeoGen.git
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
├── solver.py           # symbolic solver with our-implemented forward-search
├── target_finder.py    # Conduct reasoning, find target and create question & answer
└── ...

```

## Create Q&A for Geometry3K & PGPS9K

```bash
python main_search_public.py \
    --dataset_name geo3k # or pgps9k \
    --save_dir /path/to/your/dir \
    --num_process 12 \
    --seed 1234 \
    --debug False

```

## Synthetis Geometry Diagram and Create Q&A

```bash
python main_search_synth.py \
    --save_dir /path/to/your/dir \
    --num_process 12 \
    --seed 1234 \
    --debug False \
    --use_default_sampling_num True
    # you can disable this para and modify the sampling num in python sript.
```

## License



