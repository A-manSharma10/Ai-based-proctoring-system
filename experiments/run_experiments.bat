@echo off
echo Starting Research Experiment Pipeline...

echo.
echo Running Single-Modal Experiment...
python -m experiments.experiment_runner --mode single_modal --dataset experiments/dataset/sample_dataset.json

echo.
echo Running Multimodal Experiment...
python -m experiments.experiment_runner --mode multimodal --dataset experiments/dataset/sample_dataset.json

echo.
echo Generating Research Summary...
python -m experiments.report_generator

echo.
echo Experiments completed successfully. Results and graphs are in 'experiments/results' and 'experiments/graphs'.
pause
