from algorithms_my import PolicyGradeintAgent
import gymnasium
import time

if __name__ == '__main__':
    # env_name = "CartPole-v1"
    # env_name = "Pendulum-v1"
    # env_name = "Humanoid-v5"
    # env_name = "InvertedPendulum-v5"
    env_name = "Ant-v5"

    # num_envs = 1
    num_envs = 12
    horizon_length = 16
    batch_size = num_envs * horizon_length
    num_epochs = 5
    algorithm = 'ppo'
    num_mini_epochs = 4

    iswandb = False


    istrain = True
    isload = False
    loaddir = 'Ant-v5_10_20250725_193336_Norm obs Denorm Val'
    # loaddir = 'Ant-v5_32_20250726_000942_Norm obs Denorm Val'
    # loaddir = 'Ant-v5_32_20250728_191815_PPO Mini Epoch + Running'
    # loaddir = 'Ant-v5_1024_gae'
    # loaddir = 'Ant-v5_1024_nogae'
    # loaddir = 'Ant-v5_old'
    loadpoint = 9500
    loadpath = f"run/{loaddir}/{str(loadpoint)}.pth"


    
    save_name = "PPO Mini Epoch + Running"


    # Gymnasium Environment
    start_time = time.time()

    if istrain:
        print("Train Mode")
        if num_envs == 1:
            envs = gymnasium.make(env_name)
            envs = gymnasium.wrappers.Autoreset(envs)
        else:
            envs = gymnasium.make_vec(
                env_name,
                num_envs=num_envs,
                vectorization_mode="sync",
            )
    else:
        print("Render Mode")
        num_envs = 1
        envs = gymnasium.make(env_name, render_mode='human')
        envs = gymnasium.wrappers.Autoreset(envs)

    print(f"Make {num_envs} Env Complete")

    make_time = time.time() - start_time
    print(f"Gym Make : {make_time:.6f}sec")


    # Load
    agent =  PolicyGradeintAgent(envs, num_envs, num_epochs, horizon_length, batch_size, save_name, algorithm, num_mini_epochs, iswandb)
    if isload:
        agent.load(loadpath)


    # Train or Eval
    if istrain:
        agent.train()
    else:
        agent.eval()

    ''' add anything (e.g., visualization, learning graph, etc)'''        
