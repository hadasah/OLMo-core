## srun + python run connection error

```
g3053:52835:53467 [3] NCCL INFO [Service thread] Connection closed by localRank 0
2025-06-16 16:58:45.173	g3053:1	olmo_core.train.trainer:648	ERROR	Training failed due to:
[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:34971
2025-06-16 16:58:45.173	g3053:2	olmo_core.train.trainer:648	ERROR	Training failed due to:
[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:12994
2025-06-16 16:58:45.173	g3053:3	olmo_core.train.trainer:648	ERROR	Training failed due to:
[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:61116
2025-06-16 16:58:45.173	g3053:1	__main__:306	ERROR	Error occurred: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:34971
2025-06-16 16:58:45.173	g3053:3	__main__:306	ERROR	Error occurred: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:61116
2025-06-16 16:58:45.173	g3053:1	__main__:307	ERROR	Traceback:
2025-06-16 16:58:45.173	g3053:3	__main__:307	ERROR	Traceback:
2025-06-16 16:58:45.173	g3053:2	__main__:306	ERROR	Error occurred: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:12994
2025-06-16 16:58:45.173	g3053:2	__main__:307	ERROR	Traceback:
2025-06-16 16:58:45.345	g3053:1	__main__:308	ERROR	Traceback (most recent call last):
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/scripts/single_train_launch.py", line 303, in main
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/train/trainer.py", line 646, in fit
    self._fit_epoch()
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/train/trainer.py", line 1139, in _fit_epoch
    self.data_loader.reshuffle(self.epoch)
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/data/data_loader.py", line 486, in reshuffle
    self.build_and_save_global_indices(in_memory=in_memory)
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/data/data_loader.py", line 477, in build_and_save_global_indices
    barrier()
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/distributed/utils.py", line 173, in barrier
    dist.barrier(group)
  File "/gscratch/zlab/margsli/miniforge3/envs/olmoe-core/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/olmoe-core/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 4627, in barrier
    work.wait()
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [10.64.77.53]:34971
```

Never did figure out how to use srun (which has the very helpful kill on bad exit -K1 option). 

## Torch dynamo error

```
2025-06-16 16:44:37.316	g3076:0	__main__:306	ERROR	Error occurred: Dynamo failed to run FX node with fake tensors: call_method view(*(FakeTensor(..., device='cuda:0', size=(4, 2048, 832), dtype=torch.bfloat16,
           grad_fn=<ToCopyBackward0>), 4, 2048, -1, 75), **{}): got RuntimeError("shape '[4, 2048, -1, 75]' is invalid for input of size 6815744")

from user code:
   File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/nn/transformer/block.py", line 200, in forward
    h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/src/olmo_core/nn/attention/__init__.py", line 421, in forward
    q = q.view(B, T, -1, self.head_dim)
```

Accidentally set the wrong hidden dimension in the TransformerConfig

## Not using all gpus

Code was silently failing to use all gpus. I noticed this bc training speeds were suss, and didn't change when adding more GPUs. 
This was also visible by looking at the wandb logs for each GPU's memory and power usage (near 0 for all but GPU 0)
Also visible in logs bc all logs were from GPU 0. If all GPUs are active, the NCCL info logs should come from all GPUs. But most training logs are supposed to only go to GPU 0, so you can't tell that way.
```
g3094:44814:45294 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/CUMEM
g3094:44814:45294 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/CUMEM
g3094:44814:45294 [1] NCCL INFO Channel 02/0 : 1[1] -> 0[0] via P2P/CUMEM
g3094:44814:45294 [1] NCCL INFO Channel 03/0 : 1[1] -> 0[0] via P2P/CUMEM
g3094:44813:45295 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
g3094:44813:45295 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM
g3094:44813:45295 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM
g3094:44813:45295 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM
```

and also the model checkpoints were only `__0_{index}.distcp` files, indicating they all came from GPU 0.

