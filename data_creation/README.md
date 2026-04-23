
Create a json file that contains all the jobs (build_3dgs_pairs.py) then run with run_jobs_multi_gpu.py.

```
python build_3dgs_pairs_clean.py --scenes_root $ROOT_PATH \
        --out_root $OUT_PATH \
        --jobs_file $OUT_PATH/jobs_train.jsonl \
        --Ks 6,8,10,12 \
        --runs_per_K 3 \
        --max_iter 3000

python run_jobs_multi_gpu.py \
  --jobs_file $OUT_PATH/jobs.jsonl \
  --gpus 0,1,2,3 \
  --max_retries 1
```
