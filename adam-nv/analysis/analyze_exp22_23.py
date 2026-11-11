"""Descriptive paired-seed endpoint and trajectory analysis. Experimental unit: seed."""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
import numpy as np

METRICS = ("val_acc", "train_corrupt_noisy_label_acc", "train_corrupt_clean_label_acc", "train_clean_subset_acc")

def load(root):
    runs = defaultdict(dict)
    for file in glob.glob(os.path.join(root,"trajectory_*.csv")):
        match = re.match(r'trajectory_(pilot|frozen|adam_nv|adam)_ln([\d.]+)_seed(\d+)\.csv$',os.path.basename(file))
        if not match: continue
        method, noise, seed = match.groups()
        with open(file, newline='') as f: rows = list(csv.DictReader(f))
        if rows: runs[(float(noise), method)][int(seed)] = {int(r['epoch']):r for r in rows}
    return runs

def mean_ci(vals):
    vals = np.asarray(vals,dtype=float)
    if not len(vals): return 'missing'
    if len(vals)<2: return f'{vals.mean():+.4f} (n=1; no CI)'
    from scipy.stats import t
    margin=t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
    return f'{vals.mean():+.4f} [95% paired t CI {vals.mean()-margin:+.4f}, {vals.mean()+margin:+.4f}], n={len(vals)}'

def collect_pair(a,b,key,epoch):
    seeds=sorted(set(a).intersection(b))
    pairs=[]
    for s in seeds:
        if epoch not in a[s] or epoch not in b[s]: continue
        x,y=a[s][epoch][key],b[s][epoch][key]
        if x!='' and y!='': pairs.append((s,float(x),float(y)))
    return pairs

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dir',required=True)
    p.add_argument('--noise',type=float,default=.2)
    p.add_argument('--reference',choices=['adam_nv','adam'],default='adam_nv')
    p.add_argument('--comparison',choices=['frozen','adam','adam_nv'],default='frozen')
    p.add_argument('--output',default=None)
    args=p.parse_args()
    runs=load(args.dir)
    a,b=(runs.get((args.noise,m),{}) for m in (args.reference,args.comparison))
    if not a or not b: raise SystemExit('Missing comparison runs: check log directory, methods and noise')
    common=sorted(set(a)&set(b))
    if not common: raise SystemExit('No paired seeds; cannot compute paired differences')
    epochs=sorted(set.intersection(*(set(a[s])&set(b[s]) for s in common)))
    if not epochs: raise SystemExit('No shared epochs')
    print(f'Paired comparison {args.comparison} minus {args.reference}, label noise {args.noise}, seeds {common}')
    print('All percentages below are differences in FRACTIONS unless multiplied by 100. These CIs are exploratory and unadjusted.')
    for metric in METRICS:
        print('\n', metric)
        for e in epochs:
            pairs=collect_pair(a,b,metric,e)
            if not pairs: continue
            if e==epochs[-1] or e in (1,5,10,20,30,45,60):
                diff=np.array([y-x for _,x,y in pairs])
                print(f'  epoch {e:3d} delta pp: {mean_ci(100*diff)}')
        end_pairs=collect_pair(a,b,metric,epochs[-1])
        if end_pairs:
            print('  final paired raw values:', [(s,round(x,6),round(y,6)) for s,x,y in end_pairs])
    # Match the trajectory by observed clean/noisy training-metric levels only.
    # Descriptive interpolation; does not equalize SGD history, loss, or compute.
    out=args.output or os.path.join(args.dir, 'exp22_23_paired_trajectory.csv')
    with open(out,'w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['epoch','seed','metric','reference','comparison','delta_pp'])
        for metric in METRICS:
            for e in epochs:
                for s,x,y in collect_pair(a,b,metric,e):
                    wr.writerow([e,s,metric,x,y,100*(y-x)])
    print('\nSaved complete paired trajectories to',out)
    print('Inference caution: evaluate pairwise SEED deltas; epochs and images within a seed are not independent.')

if __name__=='__main__': main()
