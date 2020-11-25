#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de  # src
tgt=en  # tgt
lang=de-en # Language pair 
prep=preped.$lang # Prepared file location
tmp=$prep/tmp # Temp file
orig=orig # Original folder

mkdir -p $tmp $prep

echo "pre-processing train data..."
for l in $src $tgt; do
    f=inp.$lang.$l      # Input file name
    tok=out.$lang.tok.$l # Output file name

    cat $orig/$f | \
    # grep -v '<url>' | \
    # grep -v '<talkid>' | \
    # grep -v '<keywords>' | \
    # sed -e 's/<title>//g' | \
    # sed -e 's/<\/title>//g' | \
    # sed -e 's/<description>//g' | \
    # sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/out.$lang.tok $src $tgt $tmp/out.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/out.$lang.$l > $tmp/out.$lang
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do

    fname=inp.test.$lang.$l
    f=$tmp/${fname%.*}
    echo $o $f
    cat $o | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""

done


echo "creating train, valid, test..."
for l in $src $tgt; do
    # Valid set is every 23rd line 
    awk '{if (NR%23 == 0)  print $0; }' $tmp/out.$lang.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/out.$lang.$l > $tmp/train.$l

    cat $tmp/inp.test.$lang.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/BPE_CODE
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
