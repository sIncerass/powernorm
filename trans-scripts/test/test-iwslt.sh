#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
OUTPUT_PATH=$1
CKPT=$2
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
#OUTPUT_PATH=log/$PROBLEM/$ARCH\_$suffix
mkdir -p $OUTPUT_PATH

BEAM_SIZE=5
LPEN=1.0
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')
#MOSES_PATH=trans-scripts/data-preprocessing/mosesdecode
TRANS_PATH=$OUTPUT_PATH/trans
RESULT_PATH=$TRANS_PATH/$CKPT_ID
mkdir -p $RESULT_PATH

CKPT=$2
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
