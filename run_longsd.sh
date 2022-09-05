CUDA_VISIBLE_DEVICES=1 python longsd.py -f the_goat -n 2 &
CUDA_VISIBLE_DEVICES=0 python longsd.py -f the_goat -n 2 && fg