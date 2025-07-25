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
    num_envs = 10
    horizon_length = 16
    batch_size = num_envs * horizon_length
    num_epochs = 10000

    iswandb = False


    isload = True
    loaddir = 'Ant-v5_10_20250725_193336_Norm obs Denorm Val'
    loaddir = 'Ant-v5_1024_nogae'
    # loaddir = 'Ant-v5_old'
    loadpoint = 9500
    
    loadpath = f"run/{loaddir}/{str(loadpoint)}.pth"

    istrain = False
    if istrain:
        isrender = False
    else:
        isrender = True
    
    save_name = "Norm obs Denorm Val"


    # Gymnasium Environment
    start_time = time.time()
    if isrender:
        print("Render Mode")
        envs = gymnasium.make(env_name, render_mode="human")
    else:
        print("Train Mode")

        if num_envs==1:
            envs = gymnasium.make(env_name)
            print("Make 1 Env Complete")
        else:
            envs = gymnasium.make_vec(
                env_name,
                num_envs=num_envs,
                vectorization_mode="sync",
            )
            print("Make", num_envs, "Envs Complete")
    make_time = time.time() - start_time
    print(f"gym make : {make_time:.6f}sec")



    # Load
    agent =  PolicyGradeintAgent(envs, num_envs, num_epochs, horizon_length, batch_size, save_name, iswandb)
    if isload:
        agent.load(loadpath)


    # Train or Eval
    if istrain:
        agent.train()
    else:
        agent.eval()



    # env = gymnasium.wrappers.NormalizeObservation(env)
    # wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)


# No Train : step = 15 지속
# 2000 : step = 30 지속
# 10000 : 1000 지속



    
    ''' add anything (e.g., visualization, learning graph, etc)'''        
