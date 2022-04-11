# https://drive.google.com/file/d/{id}/view?usp=sharing 
# If you want to download manually, change the `{id}` with the id string below
mkdir -p data/train data/eval data/corpus

# pretrain data
gdown --id 10YIohcsXAHKFzF2L43qkxH5zYkzTw70R --output ./data/train/dl_10m.jsonl.gz
gdown --id 10YWz5WN_qJAXVCON47R1cWx2j8MScR1_ --output ./data/train/cm_10m.jsonl.gz
gzip -d ./data/train/dl_10m.jsonl.gz &
gzip -d ./data/train/cm_10m.jsonl.gz &

# training set
gdown --id 1-3fy6UcjVJLt6CW7vRp_OkWb37WMBRBR --output ./data/train/nq-train.jsonl
gdown --id 1-4BgqIfd8r-mK8cWP4nunOqowsG0xAJT --output ./data/train/nq-dev.jsonl

gdown --id 1-5ew6FNHYmauz5YoCKhnAb6wlTNpwEN6 --output ./data/train/trivia-train.jsonl
gdown --id 1-7qJY872hwoXN9bQQUtbV82BqVCUjSOA --output ./data/train/trivia-dev.jsonl

gdown --id 1-7DZ9dPTGIen7_dy4816v4r3fQ-F5h3C --output ./data/train/webq-train.jsonl
gdown --id 1-6HgRQ7ocB72rxgsaOhHIkOk176RWfau --output ./data/train/webq-dev.jsonl

# testset from dpr
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv -O ./data/eval/nq-test.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz -P ./data/eval/
gzip -d ./data/eval/trivia-test.qa.csv.gz &
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv -O ./data/eval/webq-test.qa.csv

# corpus from dpr
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -P ./data/corpus/
gzip -d ./data/corpus/psgs_w100.tsv.gz &

python -m spacy download en_core_web_sm