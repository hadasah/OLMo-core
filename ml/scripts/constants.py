"""
"""

import os

DEFAULT_DIR_PATH = "/".join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])


## Default hyperparameters for all models, and specific models. The "all" key applies to all models, and the specific model keys override the "all" key for that model.
## Defaults to chinchilla: D = 20N. Should be adjusted to D = 100N. Consider seq length 4096, global batch size 256.
MODEL_HP_DEFAULTS = {
    "all": {
        "global_batch_size": [512],
        "sequence_length": [2048],
        "train_module": {
            "optim": {
                "weight_decay": [0.1],
            },
            "scheduler": {
                "units": ["steps"],
                "warmup_steps": [2000],
            },
        },
        "trainer": {
            "max_duration": {
                "unit": ["tokens"],
            },
        },
    },
    "olmo2_ml_10M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [200_000_000],
            },
        },
    },
    "olmo2_ml_20M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [400_000_000],
            },
        },
    },
    "olmo2_ml_50M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1_000_000_000],
            },
        },
    },
    "olmo2_ml_80M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1_600_000_000],
            },
        },
    },
    "olmo2_ml_100M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [2_000_000_000],
            },
        },
    },
    "olmo2_ml_110M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [2_200_000_000],
            },
        },
    },
    "olmo2_ml_200M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [4_000_000_000],
            },
        },
    },
    "olmo2_ml_300M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [6_000_000_000],
            },
        },
    },
    "olmo2_ml_500M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [10_000_000_000],
            },
        },
    },
    # 1B = 20B tokens (D/N = 20), matching the klone 1B sweeps exactly
    # (--trainer.max_duration.value=20000000000 in their wandb-logged args).
    # No per-model lr override here: the sweep's main_grid pins lr=4e-4 for
    # every run, and the klone 1B configs confirm optim.lr == 0.0004.
    "olmo2_ml_1B": {
        "trainer": {
            "max_duration": {
                "value": [20_000_000_000],
            },
        },
    },
}

PROJECT_SPECS = {
    "margsli": {
        "DEFAULT_SAVE_PATH": os.path.join(DEFAULT_DIR_PATH, "models"),
        "DATA_WORK_DIR": "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
        "VALID_DATA_DIR": "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "moe",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "ckpt-g2",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 4,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/gscratch/zlab/snehark/OLMo-core/data/",
        "NAME_KEYS": [],
    },
    "ubuntu": {
        "DEFAULT_SAVE_PATH": os.path.join(DEFAULT_DIR_PATH, "models"),
        "DATA_WORK_DIR": os.path.join(DEFAULT_DIR_PATH, "ml/data"),
        "VALID_DATA_DIR": os.path.join(DEFAULT_DIR_PATH, "ml/data/preprocessed"),
        "WANDB_PROJECT": "data_rep_moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "olmoe-core",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "",
        "SLURM_PARTITION": "",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 1,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": os.path.join(DEFAULT_DIR_PATH, "data"),
        "NAME_KEYS": [],
    },
    # "atindra_coriander": {
    #       'DEFAULT_SAVE_PATH': '/m-coriander/coriander/atindra/models',
    #       'DATA_WORK_DIR': '/m-coriander/coriander/atindra/data/work',
    #       'VALID_DATA_DIR': '/m-coriander/coriander/atindra/data/preprocessed',
    #       "WANDB_PROJECT": "moe",
    #       "WANDB_ENTITY": "ml-moe",
    #       "CONDA_ENV_NAME": "olmoe-core",
    #       "PROJECT_DIR": DEFAULT_DIR_PATH,
    #       "SLURM_ACCOUNT": "zlab",
    #       "SLURM_PARTITION": "ckpt-g2",
    #       "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
    #       "NUM_GPUS": 3,
    #       "MODEL": [],
    #       "DATAROOT": "https://olmo-data.org/",
    #       "DATA_DIR": '/m-coriander/coriander/atindra/data',
    #       "NAME_KEYS": [],
    # },
    "atindra": {
        "DEFAULT_SAVE_PATH": "/gscratch/zlab/atindra/models",
        "DATA_WORK_DIR": "/gscratch/zlab/atindra/data/work",
        "VALID_DATA_DIR": "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "data_rep_moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "olmoe-core",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-h200",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 2,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/gscratch/zlab/atindra/data/",
        "NAME_KEYS": [],
    },
    # Marlowe (Stanford SRCC). $USER there is the SUNet id `atj10`, not `atindra`,
    # and single_train_launch.py keys PROJECT_SPECS off $USER, so this entry is what
    # makes the launcher work on that cluster at all.
    # Storage: $HOME (/users/atj10) is only 32 GB, so everything heavy lives on
    # scratch (Lustre, NOT backed up).
    # NOTE the path is /scratch/m000137-pm06, named after the ALLOCATION, not
    # /scratch/m000137, which is named after the parent project. atj10 is in group
    # marlowe-m000137-pm06 but NOT marlowe-m000137, and /scratch/m000137 is mode
    # drwxrws--- group marlowe-m000137, so that path gives Permission denied.
    # VALID_DATA_DIR must hold the v3_small_ppl_validation shards; stage them once
    # (11 files, ~27 MB total) rather than streaming them, since eval runs often.
    "atj10": {
        "DEFAULT_SAVE_PATH": "/scratch/m000137-pm06/atj10/models",
        "DATA_WORK_DIR": "/scratch/m000137-pm06/atj10/data/work",
        "VALID_DATA_DIR": "/scratch/m000137-pm06/atj10/data/preprocessed",
        "WANDB_PROJECT": "data_rep_moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "olmoe-core",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        # `batch` charges the suffixed allocation account; `preempt` must use the
        # bare `marlowe-m000137` or it bills GPU-hours for what should be free.
        "SLURM_ACCOUNT": "marlowe-m000137-pm06",
        "SLURM_PARTITION": "batch",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 4,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/scratch/m000137-pm06/atj10/data/",
        "NAME_KEYS": [],
    },
    "rohan_sanda": {
        "DEFAULT_SAVE_PATH": "/m-coriander/coriander/rohan_sanda/models",
        "DATA_WORK_DIR": "/m-coriander/coriander/rohan_sanda/OLMo-core/ml/data",
        "VALID_DATA_DIR": "/m-coriander/coriander/rohan_sanda/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "moe-rohan",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "olmo-core",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "ckpt-g2",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 1,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/m-coriander/coriander/rohan_sanda/data/",
        "NAME_KEYS": [],
    },
}

HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 4,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
        "per_gpu_batch_size": 16,
    },
    "ckpt-g2": {
        "JOBTIME": "9:00:00",
    },
    "olmo2_ml_10M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2_ml_20M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2_ml_50M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2_ml_80M": {
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 180,
        },
    },
    "olmo2_ml_100M": {
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 180,
        },
    },
    "olmo2_ml_200M": {
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 220,
        },
    },
    "olmo2_ml_300M": {
        "gpu-h200": {
            "per_gpu_batch_size": 16,
            "NUM_CPUS": 16,
            "MEM_GB": 240,
        },
    },
    "olmo2_ml_500M": {
        "gpu-l40": {
            "per_gpu_batch_size": 8,
        },
        "gpu-a40": {
            "per_gpu_batch_size": 8,
        },
        "gpu-h200": {
            "per_gpu_batch_size": 16,
            "NUM_CPUS": 16,
            "MEM_GB": 240,
        },
    },
    # Marlowe 1B settings. Keyed "all" (reachable now that train_sweep merges the
    # model-level "all" block); klone's 1B runs used 8 GPUs per job (gpu_count=8
    # in every run's wandb metadata), so NUM_GPUS=8 replicates that layout on one
    # H100 node. 47:59:59 rides just under the batch partition's 2-day cap;
    # requeue=True + load_strategy "if_available" resume from the latest
    # checkpoint if a job times out anyway. per_gpu_batch_size is deliberately
    # NOT set here: train_sweep never forwards it to the launch command (the
    # parser default of 16 is what klone ran with), and the moe64 arm's H100
    # memory override is passed explicitly in its subgrids instead.
    "olmo2_ml_1B": {
        "all": {
            "NUM_GPUS": 8,
            "MEM_GB": 800,
            "JOBTIME": "47:59:59",
        },
    },
}
