BATCH_SIZE=2000
EMBEDDING_DIM=100
NUM_EPOCHS=300
NUM_EPOCHS_PEP=300
BETA=0.8
MARGIN_KGE=4.0
MARGIN_HIER=4.0
MODEL_NAME="TransE"
LEARNING_RATE=0.001
PEP="dimension"

echo " python ./train.py --data_dir data/yago --batch_size $BATCH_SIZE  --embedding_dim $EMBEDDING_DIM \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE  \
    --beta $BETA \
    --margin_kge $MARGIN_KGE \
    --margin_hier $MARGIN_HIER \
    --model_name $MODEL_NAME \
    --pep $PEP "

python ./train.py --data_dir data/yago --batch_size $BATCH_SIZE  --embedding_dim $EMBEDDING_DIM \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE  \
    --beta $BETA \
    --margin_kge $MARGIN_KGE \
    --margin_hier $MARGIN_HIER \
    --model_name $MODEL_NAME \
    --pep $PEP