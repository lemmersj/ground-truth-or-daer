"""Performs an analysis run of a rejection model.
"""
import argparse
import os
from distutils.util import strtobool
import wandb
import generate_score_csvs
from calc_additional_error import AECalc

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--gt_csv', type=str, default=None, required=True)
PARSER.add_argument('--mturk_csv', type=str, default=None, required=True)
PARSER.add_argument('--method', type=str, default=None, required=True)
PARSER.add_argument('--weights', type=str, default=None, required=True)
PARSER.add_argument('--dir', type=str, required=True)
PARSER.add_argument('--bins', type=int, default=200)
PARSER.add_argument('--percentile', type=int, default=-1)
PARSER.add_argument("--error", action="store_true", default=False)
PARSER.add_argument("--random", type=str, default="false")
PARSER.add_argument("--blind", type=str, default="false")
ARGS = PARSER.parse_args()

ARGS.random = strtobool(ARGS.random)
ARGS.blind = strtobool(ARGS.blind)
ARGS.outfile = ARGS.dir+"/scores.csv"
ARGS.hist_test = ARGS.dir+"/overall_histogram.hist"

wandb.init(
    project="gtd-kcve", config=ARGS.__dict__,
    name=ARGS.mturk_csv+"-"+ARGS.method+"-"+str(ARGS.percentile))

os.makedirs(ARGS.dir, exist_ok=True)

# Generate the csv
generate_score_csvs.main(ARGS)
AECalc(ARGS.outfile)
