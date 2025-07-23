# python src/data_handling/build_dataset.py --config /home/rgarcia/mcs/config/config_rj_daily_casesonly.yaml
# python src/data_handling/build_dataset.py --config /home/rgarcia/mcs/config/config_rj_daily.yaml
# python src/data_handling/build_dataset.py --config /home/rgarcia/mcs/config/config_rj_weekly_casesonly.yaml
# python src/data_handling/build_dataset.py --config /home/rgarcia/mcs/config/config_rj_weekly.yaml

python src/train_pipeline.py --config config/train_rj_weekly_casesonly.yaml
python src/train_pipeline.py --config config/train_rj_daily_casesonly.yaml
python src/train_pipeline.py --config config/train_rj_daily.yaml
python src/train_pipeline.py --config config/train_rj_weekly.yaml