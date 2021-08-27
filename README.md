# Image Classification

## **Easy Start**

1. pip install

```sh
pip3 install -r requirements.txt
```

2. train, test dataset 다운로드 (압축 파일)

```sh
wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000074/data/train.tar.gz`
```

3. 압축 해제 (train, eval 디렉토리 생성됨)

```sh
tar -zxvf train.tar.gz
```

4. train 디렉토리에 train_labeled.csv 생성

```sh
wget https://boostcamp.s3.ap-northeast-2.amazonaws.com/train_labeled.csv -P ./train
```

5. main.py 실행 

```sh
python3 main.py
```





