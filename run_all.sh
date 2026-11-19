#!/usr/bin/env bash
# about 1 h 15 min on 2 cores: 400 finite element designs (14 field solutions each), validation, training, figures
set -e
python -m motorsur.dataset_fem 400   # resumes if interrupted
python -m motorsur.validate
python -m motorsur.train
python -m motorsur.plots
