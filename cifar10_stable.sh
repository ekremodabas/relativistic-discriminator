# for dcgan setup
python main.py --loss_type sgan        --batch_size 64 --spec_norm True
python main.py --loss_type rsgan       --batch_size 64 --spec_norm True
python main.py --loss_type rasgan      --batch_size 64 --spec_norm True
python main.py --loss_type hingegan    --batch_size 64 --spec_norm True
python main.py --loss_type rahingegan  --batch_size 64 --spec_norm True
python main.py --loss_type lsgan       --batch_size 64 --spec_norm True
python main.py --loss_type ralsgan     --batch_size 64 --spec_norm True
python main.py --loss_type wgan-gp     --batch_size 64 --spec_norm True
python main.py --loss_type rsgan-gp    --batch_size 64 --spec_norm True
python main.py --loss_type rasgan-gp   --batch_size 64 --spec_norm True


# for wgan gp setup
python main.py --loss_type sgan        --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type rsgan       --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type rasgan      --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type hingegan    --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type rahingegan  --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type lsgan       --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type ralsgan     --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type wgan-gp     --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
python main.py --loss_type rsgan-gp    --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5
