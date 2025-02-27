import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
#import ipdb
import os

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import torchvision
from torchvision import transforms

state_size = 29
action_size = 3
car_state_size = 9
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  #to change
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = 0    # train or not
TAU = 0.001

VISION = False
if VISION:
    state_size = 512
    # state_size = 512 + car_state_size

exp = 'models/basic_ftrs/'
if not os.path.isdir(exp):
    os.mkdir(exp)


SAVE_IMAGES = False # if set to True, will save rgb images (64,64) in current path under /saved_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OU = OU()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)


actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

transform = transforms.Compose([            #[1]
 # transforms.Resize(256),                    #[2]
 # transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

# feature extractor based on ResNet18, out_dim = 512
ftr_extractor = torchvision.models.resnet18(pretrained=True)
ftr_extractor = torch.nn.Sequential(*(list(ftr_extractor.children())[:-1]))
ftr_extractor.eval()
ftr_extractor.to(device)
#load model
print("loading model")
try:
    if train_indicator==0:
        actor.load_state_dict(torch.load(exp + 'actormodel.pth'))
        actor.eval()
        critic.load_state_dict(torch.load(exp + 'criticmodel.pth'))
        critic.eval()
        print("model load successfully")
except:
    print("cannot find the model")

#critic.apply(init_weights)
buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum')

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

#env environment
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 

all_rewards = []

for i in range(2000):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch = True)
    else:
        ob = env.reset()
    
    if not VISION:
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    else:
        # vision as input
        s_t = np.rot90(ob.img.T)
        # print(s_t.shape)
        # s_t = transform(s_t)
        s_t = torch.tensor(s_t.copy(), device=device)
        s_t = torch.unsqueeze(s_t, 0)
        s_t = s_t.permute(0,3,1,2)
        s_t = ftr_extractor(s_t.float()).squeeze()
        # c_t = np.hstack((ob.angle, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        # c_t = torch.tensor(c_t, device=device).float()
        # s_t = torch.cat([s_t, c_t])
    
    sum_rewards = 0
    for j in range(100000):
        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])
        #ipdb.set_trace() 
        # original
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        # print(s_t.shape)
        # print(torch.tensor(np.hstack((ob.angle, ob.track, ob.trackPos))).shape)
        # a_t_original = actor(s_t)


        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()
    
        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        #stochastic brake
        if random.random() <= 0.1:
            print("apply the brake")
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)
        
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0])
        
        if SAVE_IMAGES:
            # Make directory to save images
            SAVE_IMAGE_DIRECTORY_path = pathlib.Path.cwd() / "saved_images"
            SAVE_IMAGE_DIRECTORY_path.mkdir(parents=True, exist_ok=True)
            image_name = SAVE_IMAGE_DIRECTORY_path/f"_{j}.jpeg"
            im = Image.fromarray(np.rot90(ob.img.T)) #ob.img had a shape of (3, 64, 64) causing error in Image.fromarray => taking Transpose to convert to (64,64,3)
            im.save(image_name)

        if not VISION:
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        else:
            s_t1 = np.rot90(ob.img.T)
            # s_t1 = ftr_extractor(torch.unsqueeze(transform(s_t1), 0)).squeeze()
            s_t1 = torch.tensor(s_t1.copy())
            s_t1 = torch.unsqueeze(s_t1, 0)
            s_t1 = s_t1.permute(0,3,1,2)
            s_t1 = ftr_extractor(s_t1.float()).squeeze()
            # c_t1 = np.hstack((ob.angle, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            # c_t1 = torch.tensor(c_t1).float()
            # s_t1 = torch.cat([s_t1, c_t1])

        # #add to replay buffer
        # buff.add(s_t.cpu().detach().numpy(), a_t[0], r_t, s_t1.cpu().detach().numpy(), done)

        # batch = buff.getBatch(BATCH_SIZE)

        # states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()    #torch.cat(batch[0])
        # actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        # rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
        # new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
        # dones = np.asarray([e[4] for e in batch])
        # y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        
        # #use target network to calculate target_q_value
        # target_q_values = target_critic(new_states, target_actor(new_states))

        # for k in range(len(batch)):
        #     if dones[k]:
        #         y_t[k] = rewards[k]
        #     else:
        #         y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if(train_indicator):
            
            #training
            q_values = critic(states, actions)
            loss = criterion_critic(y_t, q_values)  
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)                         ##for param in critic.parameters(): param.grad.data.clamp(-1, 1)
            optimizer_critic.step()

            a_for_grad = actor(states)
            a_for_grad.requires_grad_()    #enables the requires_grad of a_for_grad
            q_values_for_grad = critic(states, a_for_grad)
            critic.zero_grad()
            q_sum = q_values_for_grad.sum()
            q_sum.backward(retain_graph=True)

            grads = torch.autograd.grad(q_sum, a_for_grad) #a_for_grad is not a leaf node  
            #grads is a tuple, while grads[0] is what we want

            #grads[0] = -grads[0]
            #print(grads)   

            act = actor(states)
            actor.zero_grad()
            act.backward(-grads[0])
            optimizer_actor.step()

            #soft update for target network
            #actor_params = list(actor.parameters())
            #critic_params = list(critic.parameters())
            print("soft updates target network")
            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)
        
        s_t = s_t1
        sum_rewards += r_t
        print("---Episode ", i , "  Action:", a_t, "  Reward:", r_t, "  Loss:", loss)

        if done:
            break

    all_rewards.append(sum_rewards)

    if np.mod(i, 3) == 0:
        if (train_indicator):
            print("saving model")
            torch.save(actor.state_dict(), exp + 'actormodel.pth')
            torch.save(critic.state_dict(), exp + 'criticmodel.pth')
            np.save(exp + 'rewards_train', np.array(all_rewards))

    
env.end()
print("Finish.")

#for param in critic.parameters(): param.grad.data.clamp(-1, 1)

