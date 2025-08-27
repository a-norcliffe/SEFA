echo "Running ACFlow test"
python -m tests.acflow_test

echo "Running acquisition test"
python -m tests.acquisition_test

echo "Running block layer test"
python -m tests.block_layer_test

echo "Running input layer test"
python -m tests.input_layer_test

echo "Running metrics test"
python -m tests.metrics_test

echo "Running models test on CPU"
python -m tests.models_cpu_test

echo "Running models test on GPU"
python -m tests.models_gpu_test

echo "Running preprocessing test"
python -m tests.preprocessing_test

echo "Running RL utils test"
python -m tests.rl_utils_test