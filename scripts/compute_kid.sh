PROJECT=./results/$1
DOMAIN=${2:-B}
mkdir -p $PROJECT/real
mkdir -p $PROJECT/fake
mv $PROJECT/test_latest/images/*fake_$DOMAIN* $PROJECT/fake
mv $PROJECT/test_latest/images/*real_$DOMAIN* $PROJECT/real
python ./scripts/kid.py --fake $PROJECT/fake --real $PROJECT/real --gpu_id 0