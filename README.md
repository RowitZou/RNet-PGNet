# RNet_DotAtt + PGNet

It is a CoQA model. 

The seq2seq part is an old version of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py/tree/c199de0fb738c33c02c76a392005e94f6b89add3) 

## Preprocessing
```bash
  python scripts/gen_pipeline_data.py --data_file data/coqa-train-v1.0.json --output_file1 data/coqa.train.pipeline.json --output_file2 data/seq2seq-train-pipeline
  python scripts/gen_pipeline_data.py --data_file data/coqa-dev-v1.0.json --output_file1 data/coqa.dev.pipeline.json --output_file2 data/seq2seq-dev-pipeline
  python seq2seq/preprocess.py -train_src data/seq2seq-train-pipeline-src.txt -train_tgt data/seq2seq-train-pipeline-tgt.txt -valid_src data/seq2seq-dev-pipeline-src.txt -valid_tgt data/seq2seq-dev-pipeline-tgt.txt -save_data data/seq2seq-pipeline -lower -dynamic_dict -src_seq_length 10000
  PYTHONPATH=seq2seq python seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-pipeline.vocab.pt -output_file data/seq2seq-pipeline.embed
```

## Training
`n_history` can be changed to {0, 1, 2, ..} or -1.
```bash
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python rc/main.py --trainset data/coqa.train.pipeline.json --devset data/coqa.dev.pipeline.json --n_history 2 --dir pipeline_models --use_elmo true --use_multi_gpu true
  python seq2seq/train.py -data data/seq2seq-pipeline -save_model pipeline_models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/seq2seq-pipeline.embed.enc.pt -pre_word_vecs_dec data/seq2seq-pipeline.embed.dec.pt -epochs 50 -gpuid 0 -seed 123
```

## Testing
```bash
  PYTHONPATH=. python rc/main.py --testset data/coqa.dev.pipeline.json --n_history 2 --pretrained pipeline_models --batch_size 16
  python scripts/gen_pipeline_for_seq2seq.py --data_file data/coqa.dev.pipeline.json --output_file pipeline_models/pipeline-seq2seq-src.txt --pred_file pipeline_models/predictions.json
  python seq2seq/translate.py -model pipeline_models/seq2seq_copy_acc_85.00_ppl_2.18_e16.pt -src pipeline_models/pipeline-seq2seq-src.txt -output pipeline_models/pred.txt -replace_unk -verbose -gpu 0
  python scripts/gen_seq2seq_output.py --data_file data/coqa-dev-v1.0.json --pred_file pipeline_models/pred.txt --output_file pipeline_models/pipeline.prediction.json
```

## Test Dataset Result

See [CoQA Board](https://stanfordnlp.github.io/coqa/)

|In-domain | Out-of-domain | Overall |
|---------|----------|-------|
|68.1|62.3|66.4|

## Codalab

You can test the trained model on [codalab](https://worksheets.codalab.org).

coqa-dev-v1.0.json uuid: 0xe25482

RNet+PGNet model uuid: 0xa8efc0

codalab command: 
```bash
cl run :coqa-dev-v1.0.json :RNet_DotAtt 'sh RNet_DotAtt/run_on_codalab.sh' --request-docker-image floydhub/pytorch:0.4.1-gpu.cuda9cudnn7-py3.34 --request-gpus 1 --request-memory 10g --request-network
```

