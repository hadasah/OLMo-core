PATH_TO_AWS_CLI=
COMMAND_PREFIX="$PATH_TO_AWS_CLI s3 cp s3://ai2-llm"
TOKENIZER=allenai/dolma2-tokenizer
DATASET_NAME=OLMoE-mix-0824

PREPROCESSED_DATA_DIR=${DATA_DIR}/${DATASET_NAME}

# foldername=preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/pes2o/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";

# foldername=preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/${TOKENIZER}
# mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
# ${COMMAND_PREFIX}/${foldername} ${PREPROCESSED_DATA_DIR}/${foldername} --recursive --exclude "*" --include "*.npy";




foldername=preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/${TOKENIZER}
mkdir -p $PREPROCESSED_DATA_DIR/${foldername}
j=00003
for i in $( seq -w 017 188 ); do
    filename=part-${i}-${j}.npy ;
    ${COMMAND_PREFIX}/${foldername}/${filename} ${PREPROCESSED_DATA_DIR}/${foldername}/${filename};
done