import os
import argparse
import glob
from loguru import logger

parser = argparse.ArgumentParser(description='')
parser.add_argument('--folder', type=str, required=True)

args = parser.parse_args()
folder = args.folder

log_path = glob.glob(os.path.join(folder, '*.log'))[0]
lines = [line.strip() for line in open(log_path).readlines()]
ceval_scores = [float(line.split(' ')[-1]) for line in lines[:-3]]

logger.info(f'ceval score: {sum(ceval_scores)/len(ceval_scores):.4}')

for line in lines[-3:]:
    assert 'Electric Industry' in line, 'Dataset need to be check'
    dataset_name, dataset_score = line.split(':')
    logger.info(f'{dataset_name} score {dataset_score}')     

