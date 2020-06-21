#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
ENC_NORM=$1
ENC_NORM_ff=$2
DEC_NORM=$3
DEC_NORM_ff=$4
suffix=$5
NUM=5

DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
OUTPUT_PATH=log/$PROBLEM/$ARCH\_$ENC_NORM\_$ENC_NORM_ff\_$DEC_NORM\_$DEC_NORM_ff\_warm$suffix
mkdir -p $OUTPUT_PATH

python train.py $DATA_PATH \
  --seed 1 \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --encoder-norm-self $ENC_NORM --decoder-norm-self $DEC_NORM \
  --encoder-norm-ff $ENC_NORM_ff --decoder-norm-ff $DEC_NORM_ff \
  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr 0.0015 --min-lr 1e-09 \
  --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096 --save-dir $OUTPUT_PATH \
  --update-freq 1 --no-progress-bar --log-interval 50 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 20 \
  --keep-last-epochs $NUM --early-stop $NUM \
  --restore-file $OUTPUT_PATH/checkpoint_best.pt \
  --criterion label_smoothed_cross_entropy \
  | tee -a $OUTPUT_PATH/train_log.txt


python scripts/average_checkpoints.py --inputs $OUTPUT_PATH --num-epoch-checkpoints $NUM --output $OUTPUT_PATH/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')
#MOSES_PATH=trans-scripts/data-preprocessing/mosesdecode
TRANS_PATH=$OUTPUT_PATH/trans
RESULT_PATH=$TRANS_PATH/$CKPT_ID
mkdir -p $RESULT_PATH

CKPT='averaged_model.pt'
echo $CKPT_ID
python generate.py \
    $DATA_PATH \
    --path $OUTPUT_PATH/$CKPT \
    --batch-size 128 \
    --beam $BEAM_SIZE \
    --lenpen $LPEN \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
    --quiet \
> $RESULT_PATH/res.txt
echo -n $CKPT_ID ""
tail -n 1 $RESULT_PATH/res.txt
