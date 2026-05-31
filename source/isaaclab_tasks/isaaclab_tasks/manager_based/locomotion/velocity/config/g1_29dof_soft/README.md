# G1 29dof locomotion on soft terrain 

### Vanilla MLP policy

```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-v1 --num_envs 4096 --headless --video

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Play-v1 --num_envs 1 --video
```

### Teacher-student distillation 

Teacher:
```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-v1 --num_envs 4096 --headless --video --agent rsl_rl_distillation_cfg_entry_point

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-Play-v1 --num_envs 1 --agent rsl_rl_distillation_cfg_entry_point
```

Student:
```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-v1 --num_envs 4096 --headless --video --agent rsl_rl_distillation_cfg_entry_point --checkpoint /path/to/teacher/policy

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-Play-v1 --num_envs 1 --agent rsl_rl_distillation_cfg_entry_point
```

