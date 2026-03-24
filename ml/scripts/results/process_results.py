import pandas as pd
import re

MODEL_PARAMS = {
    '10M': 14456400,
    '20M': 26882064,
    '50M': 65549440,
    '100M': 132927808,
    '200M': 260945440,
    '400M': 530889120,
    '1_0B': 1342261248,
}

df = pd.read_csv("/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/scripts/results/results.csv")

pattern = "202\d_\d\d_\d\d-\d\d_\d\d_\d\d_(\w+)_olmo2_(\d+M|1_0B)_e(.+)x(.+)[ec](\d+,\d+|\d+)(?:_(.+)gen|)(?:_lr=(.+)|)?"

df['regex'] = df['Name'].apply(lambda x: re.match(pattern, x))

df['model_size'] = df['regex'].apply(lambda x: x.group(2) if x else None)
df['param_count'] = df['model_size'].apply(lambda x: MODEL_PARAMS.get(x))
df['Group'] = df['Group'].apply(lambda x: 'MoE heterogeneous' if x == 'HetMoE' else 'Dense' if x == 'dense' else x)

df['train_module.scheduler._CLASS_'] = df['train_module.scheduler._CLASS_'].apply(lambda x: x.replace('olmo_core.optim.scheduler.', ''))

df['moe_group_total_experts'] = df['regex'].apply(lambda x: x.group(3) if x else None)
df['moe_group_1_total_experts'] = df['moe_group_total_experts'].apply(lambda x: x.split(',')[0] if x else 0)
df['moe_group_2_total_experts'] = df['moe_group_total_experts'].apply(lambda x: x.split(',')[1] if x and ',' in x else 0)

df['moe_group_dims'] = df['regex'].apply(lambda x: x.group(4) if x else None)
df['moe_group_1_dim'] = df['moe_group_dims'].apply(lambda x: x.split(',')[0] if x else 0)
df['moe_group_2_dim'] = df['moe_group_dims'].apply(lambda x: x.split(',')[1] if x and ',' in x else 0)

df['moe_group_active_experts'] = df['regex'].apply(lambda x: x.group(5) if x else None)
df['moe_group_1_active_experts'] = df['moe_group_active_experts'].apply(lambda x: x.split(',')[0] if x else 0)
df['moe_group_2_active_experts'] = df['moe_group_active_experts'].apply(lambda x: x.split(',')[1] if x and ',' in x else 0)

df['moe_generalist'] = df['regex'].apply(lambda x: x.group(6) if x else 0)
df['moe_generalist_dim'] = df['moe_generalist'].apply(lambda x: float(x) if x and x!='no' else 0)

df['moe_expert_total_dim'] = df.apply(lambda row: row.moe_group_1_total_experts * row.moe_group_1_dim + row.moe_group_2_total_experts * row.moe_group_2_dim, axis=1)
df['moe_group_1_total_dim'] = df.apply(lambda row: row.moe_group_1_total_experts * row.moe_group_1_dim, axis=1)
df['moe_group_2_total_dim'] = df.apply(lambda row: row.moe_group_2_total_experts * row.moe_group_2_dim, axis=1)
df['moe_total_dim'] = df.apply(lambda row: row.moe_group_1_total_experts * row.moe_group_1_dim + row.moe_group_2_total_experts * row.moe_group_2_dim + row.moe_generalist_dim, axis=1)


df = df[df['State'] == 'finished']

df = df.sort_values(
    by=[
        'param_count', 
        'train_module.scheduler._CLASS_',
        'train_module.optim.lr',
        'Group',
        'model.block.feed_forward_moe.name',
        'model.block.feed_forward_moe.lb_loss_weight',
        'moe_generalist_dim',
        'moe_group_1_dim',
        'moe_group_2_dim',
        'moe_group_1_total_experts',
        'moe_group_2_total_experts',
        ], 
    ascending=[
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
)

cols = ['Name',
        'model_size', 
        'train_module.scheduler._CLASS_',
        'train_module.optim.lr',
        'Group',
        'model.block.feed_forward_moe.name',
        'model.block.feed_forward_moe.lb_loss_weight',
        'moe_generalist_dim',
        'moe_group_1_dim',
        'moe_group_1_total_experts',
        'moe_group_1_active_experts',
        'moe_group_2_dim',
        'moe_group_2_total_experts',
        'moe_group_2_active_experts',
        ]
cols += [col for col in df.columns if 'eval/' in col or 'train/' in col]

df.to_csv(
    "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/scripts/results/results_processed.csv",
    columns=cols, 
    index=False
)
