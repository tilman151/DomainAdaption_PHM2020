cd ../src || exit
python run.py -e 125 -r 8 -d cuda:0 -s 42 -w 0 ./configs/cmapss/cmapss_one_baseline.json
python run.py -e 125 -r 8 -d cuda:0 -s 42 -w 0 ./configs/cmapss/cmapss_two_baseline.json
python run.py -e 125 -r 8 -d cuda:0 -s 42 -w 0 ./configs/cmapss/cmapss_three_baseline.json
python run.py -e 125 -r 8 -d cuda:0 -s 42 -w 0 ./configs/cmapss/cmapss_four_baseline.json
