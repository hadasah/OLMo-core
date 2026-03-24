"""
"""

import os

#  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])

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
    "olmo2b_10M": {
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [200000000],
            },
        },
    },
    "olmo2b_20M": { #actually 20M params
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [400000000],
            },
        },
    },
    "olmo2b_50M": { #actually 20M params
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1000000000],
            },
        },
    },
    "olmo2b_500M": { #actually 20M params
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [10000000000],
            },
        },
    },
    "olmo2_5M": { # actually 40M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                # "value": [500000000],
                "value": [500000000],
            },
        },
    },
    "olmo2_10M": { # actually 80M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1000000000],
                # "value": [1600000000],
            },
        },
    },
    "olmo2_20M": { # actually 110M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [2000000000],
            },
        },
    },
    "olmo2_50M": { # actualy 200M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [5000000000],
            },
        },
    },
    "olmo2_100M": { # actually 300M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [10000000000],
            },
        },
    },
    "olmo2_200M": { # actually 500M
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [10000000000],
            },
        },
    },
    "olmo2_400M": {
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [40000000000],
            },
        },
    },
    "olmo2_1_0B": {
        "train_module": {
            "optim": {
                # "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [100000000000],
            },
        },
    },
}

PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        'DATA_WORK_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
        # 'DATA_WORK_DIR': os.path.join(DEFAULT_DIR_PATH, 'ml/data'),
        # 'VALID_DATA_DIR': os.path.join(DEFAULT_DIR_PATH, "ml/data/preprocessed"),
        'VALID_DATA_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "olmoe-core",
        "BASH_SETUP_FILE": "~/.olmoe-core.sh",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-a40,gpu-l40",
        "TRAIN_COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "EVAL_COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_eval_launch.py",
        "NUM_GPUS": 4,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        # "DATAROOT": "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
        "DATA_DIR": "/gscratch/zlab/snehark/OLMo-core/data/",
        "NAME_KEYS": [],
    },
    "snehark": {
        ## TODO @snehark: Update these paths to your local setup
    }
}

BATCH_SIZE = 512
JOB_SPEC_KEYS = ["per_gpu_batch_size"]
HARDWARE_SPECS_DICT = {
    "all": {
        # "NUM_GPUS": 4,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
        "per_gpu_batch_size": 16,
    },
    "ckpt-g2": {
        "JOBTIME": "9:00:00",
    },
    "olmo2b_10M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2b_20M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2b_50M": {
        "all": {
            "per_gpu_batch_size": 32,
        },
    },
    "olmo2b_500M": {
        "gpu-h200": {
            "per_gpu_batch_size": 8,
            "NUM_CPUS": 16,
            "MEM_GB": 2200,
            # use 8 GPUs
        }, 
    },
    "olmo2_5M": {
    },
    "olmo2_10M": { 
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 200,
        },
    },
    "olmo2_20M": { 
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 200,
        },
    },
    "olmo2_50M": { 
        "all": {
            "per_gpu_batch_size": 16,
            "MEM_GB": 200,
        },
    },
    "olmo2_100M": {
        "all": {
            "per_gpu_batch_size": 8,
            "MEM_GB": 220,
        },
        "gpu-h200": {
            "per_gpu_batch_size": 32,
            "NUM_CPUS": 16,
            "MEM_GB": 800,
        }, 
    },
    # "olmo2_200M": { 
    #     "gpu-l40": {
    #         "per_gpu_batch_size": 8,
    #     }, 
    #     "gpu-a40": {
    #         "per_gpu_batch_size": 8,
    #     }, 
    #     "gpu-h200": {
    #         "per_gpu_batch_size": 16,
    #         "NUM_CPUS": 16,
    #         "MEM_GB": 240,
    #     }, 
    # },
    # "olmo2_400M": { 
    #     "gpu-l40": {
    #         "per_gpu_batch_size": 8,
    #     }, 
    #     "gpu-a40": {
    #         "per_gpu_batch_size": 8,
    #     }, 
    #     "gpu-h200": {
    #         "per_gpu_batch_size": 16,
    #         "NUM_CPUS": 16,
    #         "MEM_GB": 800,
    #         # use 4
    #     }, 
    # },
    "olmo2_1_0B": { 
        "gpu-l40": {
            "per_gpu_batch_size": 4,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 4,
        }, 
        "gpu-h200": {
            "per_gpu_batch_size": 4,
            "NUM_CPUS": 16,
            "MEM_GB": 2200,
            # use 8
        }, 
    }
}
