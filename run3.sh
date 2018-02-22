if [ ! -d "processeddata" ]; then
  python preprocess.py
fi
mkdir -p log;
python train.py 8 3 >> "$(date +"log/%F_%H_%M_%S.txt")" 2>&1