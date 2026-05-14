# 1.指定动态链接器查找共享库（.so 文件）的额外路径
```bash
export LD_LIBRARY_PATH=/home/hithcat/miniconda3/envs/fangzhen/lib:$LD_LIBRARY_PATH
```

# 2.训练
```bash
python3 ./legged_gym/legged_gym/scripts/train.py --task=elspider_lidar_tunnel --num_envs=4 --max_iterations=100

python3 ./legged_gym/legged_gym/scripts/train.py --task=elspider_lidar_tunnel --num_envs=4 --max_iterations=100 --resume --load_run v1 --checkpoint -1

python3 ./legged_gym/legged_gym/scripts/train.py --task=elspider_lidar_confined --num_envs=500 --max_iterations=300 --headless

python3 ./legged_gym/legged_gym/scripts/train.py --task=elspider_lidar_tunnel --num_envs=500 --max_iterations=300 --headless

python3 ./legged_gym/legged_gym/scripts/train.py --task=elspider_lidar_flat_same_dim --num_envs=500 --max_iterations=300 --headless
```

# 3.测试
```bash
python3 ./legged_gym/legged_gym/scripts/play.py \
        --task elspider_lidar_flat_same_dim \
        --num_envs 1 \
        --load_run v1 \
        --checkpoint -1
```

# 4.训练参数设置
| 参数                  | 类型   | 说明                                                           |
| ------------------- | ---- | ------------------------------------------------------------ |
| `--task`            | str  | 任务名称（对应 task_registry 中注册的任务），如 `elspider_air_rough` |
| `--resume`          | bool | 是否从已有 checkpoint 恢复训练                                        |
| `--experiment_name` | str  | 实验名称（日志、模型保存的顶级目录）                                           |
| `--run_name`        | str  | 单次运行名称（experiment 下的子目录）                                     |
| `--load_run`        | str  | resume 时加载的运行名；`-1` 表示最近一次                                   |
| `--checkpoint`      | int  | resume 时加载的 checkpoint 编号；`-1` 表示最新                          |
| `--headless`        | bool | 无窗口模式运行（推荐训练时开启）                                             |
| `--horovod`         | bool | 是否使用 Horovod 进行多 GPU 分布式训练                                   |
| `--rl_device`       | str  | RL 算法使用的设备，如 `cuda:0`、`cpu`                                  |
| `--num_envs`        | int  | 并行环境数量（覆盖 config 中的设置）                                       |
| `--seed`            | int  | 随机种子                                                         |
| `--max_iterations`  | int  | 最大训练迭代次数（覆盖 config 中设置）                                      |


# 5.训练参数示例
```bash
python3 /home/hithcat/Code/DaChuang/final/OmniPerception/legged_gym/legged_gym/scripts/train.py \
    --task elspider_air_rough \
    --sim_device cuda \
    --rl_device cuda \
    --graphics_device_id 0 \
    --num_envs 5 \
    --max_iterations 10

# 射线投射版本
python3 /home/hithcat/Code/DaChuang/final/OmniPerception/legged_gym/legged_gym/scripts/train.py \
    --task elspider_air_rough_raycast \
    --sim_device cuda \
    --rl_device cuda \
    --graphics_device_id 0 \
    --num_envs 5 \
    --max_iterations 10
```

# Tensorboard
```bash
tensorboard --logdir /home/hithcat/Code/DaChuang/LidarSpider/legged_gym/logs/elspider_air_rough_lidar/v1
```
