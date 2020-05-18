# for cat 64x64 experiments, first run extract.sh in datasets

python main.py --model_type sgan        --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type rsgan       --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type rasgan      --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type hingegan    --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type rahingegan  --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type lsgan       --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type ralsgan     --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type rsgan-gp    --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
python main.py --model_type rasgan-gp   --batch_size 64 --dataset cat  --model dcgan_64 --fid_iter 10000 --save_model 10000
