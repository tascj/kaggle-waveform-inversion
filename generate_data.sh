python generate_data.py --ann-file ../artifacts/dtrainval_orig_6ch.parquet

for type in 1 2; do
    for offset in 0 8; do
        python generate_data.py --ann-file ../artifacts/dtrainval_blend_type${type}_${offset}.parquet
    done
done
