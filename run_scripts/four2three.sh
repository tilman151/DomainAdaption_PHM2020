# JAN run
python vary.py \
-e 125 \
-d cuda:0 \
-r 3 \
-s 42 \
-w 0 \
./configs/cmapss/four2three/cmapss_four2three_jan.json \
dataset.parameters.target_dataset.parameters.percent_broken=[0.0,0.2,0.4,0.6,0.8] \
dataset.parameters.target_dataset.parameters.percent_fail_runs=[1.0,0.8,0.6,0.4,0.2]
# DAAN run
python vary.py \
-e 125 \
-d cuda:0 \
-r 3 \
-s 42 \
-w 0 \
./configs/cmapss/four2three/cmapss_four2three_dadv.json \
dataset.parameters.target_dataset.parameters.percent_broken=[0.0,0.2,0.4,0.6,0.8] \
dataset.parameters.target_dataset.parameters.percent_fail_runs=[1.0,0.8,0.6,0.4,0.2]