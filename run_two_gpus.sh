if [[ ! -v OPENAI_TOKEN ]]; then
    echo "OPENAI_TOKEN is not set"
elif [[ -z "$OPENAI_TOKEN" ]]; then
    echo "OPENAI_TOKEN is set to empty string"
else
    CUDA_VISIBLE_DEVICES=1 python run.py "$@" &
    CUDA_VISIBLE_DEVICES=0 python run.py "$@" && fg
fi


