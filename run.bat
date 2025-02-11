python -m training.main ^
         --train-data="pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -" ^
         --train-num-samples 3000000 ^
         --val-data="pipe:aws s3 cp s3://s-mas/cc3m/{00330..00331}.tar -" ^
         --val-num-samples 10000 ^
         --dataset-type webdataset ^
         --batch-size 256 ^
         --warmup 2000 ^
         --epochs 10 ^
         --lr 5e-4 ^
         --precision amp ^
         --workers 6 ^
         --model "Clip" ^
         --lock-text ^
         --lock-text-unlocked-layers 10 ^
         --name "10_unfrozen1" ^
         --report-to "tensorboard" ^