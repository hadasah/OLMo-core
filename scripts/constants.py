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
}
