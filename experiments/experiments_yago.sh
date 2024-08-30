python ./train.py --data_dir data/yago --batch_size 2000  --embedding_dim 200 --num_epochs 500 \
    --learning_rate 0.001 \
    --beta 0.5 \
    --margin_kge 6.0 \
    --margin_hier 6.0 \
    --model_name TransE