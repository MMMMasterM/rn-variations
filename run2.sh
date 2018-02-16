if [ ! -d "processeddata" ]; then
  python preprocess.py
fi
mkdir -p log;
python train.py 5 >> "log/%F_%H_%M_%S.txt" 2>&1