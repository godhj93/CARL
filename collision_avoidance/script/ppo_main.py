# PPO main
# coded by St.Watermelon
## PPO 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
from ppo_learn import PPOagent

import os
from airgym.envs.drone_env import AirSimDroneEnv

def main():

    max_episode_num = 100000000  # 최대 에피소드 설정
    
    env = AirSimDroneEnv(ip_address='127.0.0.1') 

    agent = PPOagent(env) # PPO 에이전트 객체
    
    # 학습 진행
    agent.train(max_episode_num)

    # # 학습 결과 도시
    # agent.plot_result()

if __name__=="__main__":
    main()
