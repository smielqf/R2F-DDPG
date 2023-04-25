import enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def plot_all(path, num_agent, update_gap):
    total_rewards = np.load(os.path.join(path, "total_rewards.npy"))
    actor_losses = np.load(os.path.join(path, "actor_losses.npy"))
    critic_losses = np.load(os.path.join(path, "critic_losses.npy"))
    q_values = np.load(os.path.join(path, "q_values.npy"))


    for j in range(num_agent):
        plt.clf()
        plt.figure(figsize=(14, 10))
        plt.subplot(2,2,1)
        data = [np.mean(total_rewards[j][i:i+update_gap]) for i in range(0, len(total_rewards[j]), update_gap)]
        plt.plot(data)
        plt.title('Rewards of agent {}'.format(j), fontsize=20)
        plt.ylim((-100, 0))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.subplot(2,2,2)
        new_critic_losses = []
        for ele in critic_losses[j]:
            if ele != .0:
                new_critic_losses.append(np.log(ele))
        plt.plot(new_critic_losses)
        plt.title('Q loss of agent {}'.format(j), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.subplot(2,2,3)
        new_actor_losses = []
        temp = actor_losses[j]
        for index, ele in enumerate(temp):
            if ele != .0:
                new_actor_losses.append(ele)
        plt.plot(new_actor_losses)
        plt.title('Actor Loss of agent {}'.format(j), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.subplot(2,2,4)
        new_q_values = []
        for ele in q_values[j]:
            if ele != .0:
                new_q_values.append(ele)
        plt.plot(new_q_values)
        plt.title('Q-values of agent {}'.format(j), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        
        file_name = os.path.join(path, 'agent_{}'.format(j))
        plt.savefig(file_name)

    
    

def plot_jointed_all(path, num_agent, update_gap):
    total_rewards = np.load(os.path.join(path, "total_rewards.npy"))

    
    plt.clf()
    plt.figure(figsize=(16, 10))
    # plt.subplot(2,2,1)
    for j in range(num_agent):
        data = [np.mean(total_rewards[j][i:i+update_gap]) for i in range(0, len(total_rewards[j]), update_gap)]
        plt.plot(data, label='Agent {}'.format(j))
    plt.legend(fontsize=35)
    # plt.title('Rewards', fontsize=20)
    plt.ylim((-100, 0))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.ylabel("Mean Rewards", fontsize=35)
    plt.xlabel("Episodes", fontsize=35)
    
    file_name = os.path.join(path, 'Joint.png')
    plt.savefig(file_name)

    
    

def plot_ave_jointed_all(path_list, save_dir, num_agent, update_gap, load=False, log=False):
    seed_range = 5
    total_rewards = {}
    actor_losses = {}
    critic_losses = {}
    q_values = {}
    para_norms = {}
    grad_norms = {}
    para_vec_norms = {}
    grad_vec_norms = {}
    first_para_norms = {}
    second_para_norms = {}
    first_grad_norms = {}
    second_grad_norms = {}
    first_para_vec_norms = {}
    second_para_vec_norms = {}
    first_grad_vec_norms = {}
    second_grad_vec_norms = {}
    rank_change = {}

    for i, path in enumerate(path_list):
        total_rewards['seed_{}'.format(i)] = np.load(os.path.join(path, "total_rewards.npy"))
        actor_losses['seed_{}'.format(i)] = np.load(os.path.join(path, "actor_losses.npy"))
        critic_losses['seed_{}'.format(i)] = np.load(os.path.join(path, "critic_losses.npy"))
        q_values['seed_{}'.format(i)] = np.load(os.path.join(path, "q_values.npy"))
        

    
    plt.clf()
    plt.figure(figsize=(16, 10))
    # plt.subplot(2,2,1)
    for j in range(num_agent):
        data_list = []
        for seed in range(seed_range):
            _total_rewards = total_rewards['seed_{}'.format(seed)]
            data = [np.mean(_total_rewards[j][i:i+update_gap]) for i in range(0, len(_total_rewards[j]), update_gap)]
            data_list.append(data)
        data_array = np.asarray(data_list)
        min_data = np.min(data_array, axis=0, keepdims=False)
        max_data = np.max(data_array, axis=0, keepdims=False)
        mean_data = np.mean(data_array, axis=0,keepdims=False)
        # plt.plot(data, label='Agent {}'.format(j))
        plt.plot([x*update_gap for x in range(1, len(data)+1)], mean_data, label='Agent {}'.format(j+1), linewidth=3)
        plt.fill_between([x*update_gap for x in range(1, len(data)+1)], min_data, max_data, alpha=0.3)
    plt.legend(fontsize=35)
    # plt.title('Rewards', fontsize=20)
    plt.ylim((-250, 0))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.ylabel("Mean Rewards", fontsize=35)
    plt.xlabel("Episodes", fontsize=35)
    plt.grid()

    
    file_name = os.path.join(save_dir, 'Joint.png')
    plt.savefig(file_name)




def plot_ablation_study(path_list, save_dir, num_agent, update_gap):
    seed_range = 5
    total_rewards = {}
    actor_losses = {}
    critic_losses = {}
    q_values = {}


    for i, path in enumerate(path_list):
        total_rewards['seed_{}'.format(i)] = np.load(os.path.join(path, "total_rewards.npy"))
        actor_losses['seed_{}'.format(i)] = np.load(os.path.join(path, "actor_losses.npy"))
        critic_losses['seed_{}'.format(i)] = np.load(os.path.join(path, "critic_losses.npy"))
        q_values['seed_{}'.format(i)] = np.load(os.path.join(path, "q_values.npy"))

    plt.clf()
    plt.figure(figsize=(16, 10))
    # plt.subplot(2,2,1)
    for j in range(num_agent):
        data_list = []
        for seed in range(seed_range):
            _total_rewards = total_rewards['seed_{}'.format(seed)]
            data = [np.mean(_total_rewards[j][i:i+update_gap]) for i in range(0, len(_total_rewards[j]), update_gap)]
            data_list.append(data)
        data_array = np.asarray(data_list)
        min_data = np.min(data_array, axis=0, keepdims=False)
        max_data = np.max(data_array, axis=0, keepdims=False)
        mean_data = np.mean(data_array, axis=0,keepdims=False)
        # plt.plot(data, label='Agent {}'.format(j))
        plt.plot([x*update_gap for x in range(1, len(data)+1)], mean_data, label='Agent {}'.format(j+1), linewidth=3)
        plt.fill_between([x*update_gap for x in range(1, len(data)+1)], min_data, max_data, alpha=0.3)
    plt.legend(fontsize=35)
    # plt.title('Rewards', fontsize=20)
    plt.ylim((-100, 0))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.ylabel("Mean Rewards", fontsize=35)
    plt.xlabel("Episodes", fontsize=35)
    plt.grid()
    
    file_name = os.path.join(save_dir, 'ablation.pdf')
    plt.savefig(file_name)


if __name__ == '__main__':
    num_agent = 8
    attack_modes = ['gaussian', 'sign-flip',]
    num_malics = [1,]
    aggregates = ['mean', 'median', 'geometry_median']
    attack_modes = ['normal',]
    num_malics = [0, ]
    aggregates = ['mean',]
    # attack_modes = ['gaussian',]
    # num_malics = [0, ]
    # aggregates = ['median', 'geometry_median']
    # attack_modes = ['orthogonal',]
    # num_malics = [1,]
    # aggregates = ['mean', 'median', 'geometry_median']
    root_path = os.path.join("./figures", 'eight_agents')

    
    for i in tqdm(range(len(attack_modes))):
        attack_mode = attack_modes[i]            
        for num_malic in num_malics:
            num_malicious = "num_malic_{}".format(num_malic)
            for agg in aggregates:
                path_list = []
                for seed in range(5):
                    seed_path = os.path.join(root_path, 'seed_{}'.format(seed))
                    attack_path = os.path.join(seed_path, attack_mode)
                    num_malic_path = os.path.join(attack_path, num_malicious)
                    agg_path = os.path.join(num_malic_path, agg)
                    path_list.append(agg_path)
                save_dir = os.path.join(root_path, attack_mode)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_dir = os.path.join(save_dir, num_malicious)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_dir = os.path.join(save_dir, agg)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plot_ave_jointed_all(path_list, save_dir, num_agent, 10)
                plot_ablation_study(path_list, save_dir, num_agent, 10)