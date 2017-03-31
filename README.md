# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## My Summary:

### 1) Use Byte-Pair Encoding (BPE) to reduce vocabulary size

I use `https://github.com/rsennrich/subword-nmt`.

### 2) Use Bidirectional RNNs (`-brnn`)

### 3) Result: 
       	       
After training on 100k Chinese-English sentence pairs,
I got a BLEU score of 29.0 on NIST 06 newswire (dev set),
and a BLEU score of 25.2 on NIST 08 newswire (test set).

## Steps:

### 0) Data: Chinese-English.

#### Training set: I use a small training set of 100,000 Chinese-English sentence pairs.

| Language | Sentences | Vocabulary | Tokens |
|--------:|--------:|----------:| ------:|
| Chinese  | 100,000   | 49,424     | 2,743,464 |
| English  | 100,000   | 43,588     | 3,487,249 |

#### Development (held-out) set: NIST 06 newswire (616 sentences), 4 references.

#### Test set: NIST 08 newswire (691 sentences), 4 references.

### 1) Byte-Pair Encoding preparation.

Learn BPE:

```
cat data/train100k.zh | ./subword-nmt/learn_bpe.py > data-bpe/bpe.zh 
cat data/train100k.en | ./subword-nmt/learn_bpe.py > data-bpe/bpe.en
```

Apply BPE to training data:

```
cat data/train100k.zh | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.zh > data-bpe/train100k.zh.bpe 

cat data/train100k.en | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.en > data-bpe/train100k.en.bpe 
```

Apply BPE to dev set:

```
cat data/dev_06.zh | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.zh > data-bpe/dev_06.zh.bpe 

for i in `seq 0 3`; do cat data/dev_06.en.$i | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.en > data-bpe/dev_06.en.$i.bpe; done 
```

Apply BPE to test set:

```
cat data/test_08.zh | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.zh > data-bpe/test_08.zh.bpe 

for i in `seq 0 3`; do cat data/test_08.en.$i | ./subword-nmt/apply_bpe.py -c data-bpe/bpe.en > data-bpe/test_08.en.$i.bpe; done 
```

### 2) Preprocess the data.

Length limit: 80 (prune all sentences longer than 80 words after BPE).

```
python preprocess.py -train_src data-bpe/train100k.zh.bpe -train_tgt data-bpe/train100k.en.bpe -valid_src data-bpe/dev_06.zh.bpe -valid_tgt data-bpe/dev_06.en.0.bpe -seq_length 80 -save_data data-bpe/len80
```

### 3) Train the model (with `-brnn` and 30 epochs).

```
nohup python train.py -data data-bpe/len80.train.pt -save_model model.bi -gpus 1 -brnn -epochs 30 &>train-bpe.log.bi
```

### 4) Translate sentences in dev and test sets.

```
python translate.py -model models/model.bi_acc_0.00_ppl_19.03_e19.pt -src data-bpe/dev_06.zh.bpe -gpu 1 -output dev_06.out.e19.bpe.bi

python translate.py -model models/model.bi_acc_0.00_ppl_19.03_e19.pt -src data-bpe/test_08.zh.bpe -gpu 1 -output test_08.out.e19.bpe.bi
```


### 5) Evaluate BLEU score (unbpe first).

Dev:

```
cat dev_06.out.e19.bpe.bi | sed -e 's/@@ //g'| perl multi-bleu.perl data/dev_06.en.*
```

Result:

```
BLEU= 28.96, 67.1/38.2/21.9/12.6 (BP= 1.000, ratio= 1.000, hyp_len= 22833, ref_len= 22838
```

Test:

```
cat test_08.out.e19.bpe.bi| sed -e 's/@@ //g'| perl multi-bleu.perl data/test_08.en.*
```

Result:

```
BLEU= 25.17, 63.0/33.4/18.6/10.4 (BP= 0.995, ratio= 0.995, hyp_len= 22567, ref_len= 22678)```

