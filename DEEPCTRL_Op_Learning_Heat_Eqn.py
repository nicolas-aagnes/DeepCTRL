import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset, DataLoader
import io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test():
    SolnTens = np.load("Soln.npy")
    BC = np.load("BC.npy")
    x = np.linspace(-5,5,SolnTens.shape[0])
    t = np.linspace(0,3.14,SolnTens.shape[1])
    for i in range(5):
        T,X = np.meshgrid(t,x)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, T, SolnTens[:,:,90+i], 50, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('U(x,t)')
        plt.show()

class PDE_Dataset(Dataset):
    """PDE dataset."""

    def __init__(self, BC_file, Soln_file):
        """
        Args:
            BC_file (string): Path to .npy file with a Boundary condition tensor.
            Soln_file (string): Path to .npy file with a Grid solution tensor.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.SolnTens = torch.tensor(np.load(Soln_file)).float().to(device)
        self.BCTens = torch.tensor(np.load(BC_file)).float().to(device)

    def __len__(self):
        return int(self.SolnTens.shape[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        soln = self.SolnTens[:,:,idx]
        BC = self.BCTens[:,idx]
        sample = {"x": BC,"y":soln}

        return sample



class Net(nn.Module):
    def __init__(self,M_x,M_t):
        super().__init__()
        #Input shape (Bz,M_t)
        #Output shape (Bz,M_x,M_t)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5),padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5,5),padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5,5),padding=2)
        self.fc3 = nn.Linear(120,2* M_t * M_x*4)
        self.fc2 = nn.Linear(84,120)
        self.fc1 = nn.Linear(4*M_t,84)
        self.M_x = M_x
        self.M_t = M_t

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view((-1,2,self.M_x*2,self.M_t*2))
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.squeeze(x,dim=1)

        return x

class ruleNet(nn.Module):

    def __init__(self, M_t):
        super().__init__()

        self.fc3 = nn.Linear(120, M_t*2)
        self.fc2 = nn.Linear(84, 120)
        self.fc1 = nn.Linear(M_t, 84)
        self.M_t = M_t


    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class taskNet(nn.Module):

    def __init__(self, M_t):
        super().__init__()

        self.fc3 = nn.Linear(120, M_t*2)
        self.fc2 = nn.Linear(84, 120)
        self.fc1 = nn.Linear(M_t, 84)
        self.M_t = M_t


    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def pdeloss(outputs,lam):

    diffT = torch.diff(outputs,n=1,dim=-1,prepend =torch.unsqueeze(outputs[:,:,0],dim=2))
    secdiffX = torch.diff(torch.diff(outputs,n=1,dim=-2,prepend =torch.unsqueeze(outputs[:,0,:],dim=1),
                          append=torch.unsqueeze(outputs[:,-1,:],dim=1 )),n=1,dim=-2)

    criterion = nn.MSELoss()
    pdelossterm = criterion(diffT,lam * secdiffX)
    return pdelossterm

def pdeloss2(outputs,lam):
    diffT = torch.diff(outputs, n=1, dim=-1)[:,1:-1]

    secdiffX = torch.diff(torch.diff(outputs, n=1, dim=-2), n=1, dim=-2)[:,:,:-1]
    criterion = nn.MSELoss()
    pdelossterm = criterion(diffT, lam * secdiffX)
    return pdelossterm



def trainer(net,tasknet,rulenet,trainloader,testloader,args,verbose=True):
    criterion = nn.MSELoss()
    optimizerCNN = optim.Adam(net.parameters(), lr=0.001)
    optimizerTask = optim.Adam(tasknet.parameters(), lr=0.001)
    optimizerRule = optim.Adam(rulenet.parameters(), lr=0.001)

    Train_Loss = []
    Test_Loss = []
    best_loss = 1000
    p = 100
    epochs = args["epochs"]
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_r_loss = 0.0
        running_t_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            BC = data["x"]
            Soln = data["y"]

            # zero the parameter gradients
            optimizerCNN.zero_grad()
            optimizerTask.zero_grad()
            optimizerRule.zero_grad()

            # forward + backward + optimize
            taskOut = tasknet(BC)
            ruleOut = rulenet(BC)
            alp = torch.unsqueeze(torch.tensor(np.random.beta(0.1,0.1,size=taskOut.shape[0]),requires_grad=False),dim=1).to(args["device"])
            #alp =torch.unsqueeze(torch.tensor([0.5]*taskOut.shape[0],requires_grad=False),dim=1).to(args["device"])
            sqalp = torch.sqrt(alp)
            msqalp = 1-sqalp
            z = torch.cat((taskOut*alp,ruleOut*(1-alp)),dim=-1)

            outputs = net(z)

            loss_t = criterion(outputs[:,0,:]*sqalp, Soln[:,0,:]*sqalp)\
                     +criterion(outputs[:,-1,:]*sqalp, Soln[:,-1,:]*sqalp)\
                     +criterion(outputs[:,:,0]*sqalp, Soln[:,:,0]*sqalp)\
                     +criterion(outputs[:,:,-1]*sqalp, Soln[:,:,-1]*sqalp)
            loss_r = pdeloss2(outputs*torch.unsqueeze(msqalp,dim=2),args["lam"])
            loss = loss_t+p*loss_r
            loss.backward()
            optimizerCNN.step()
            optimizerTask.step()
            optimizerRule.step()

            # print statistics
            running_loss += loss.item()
            running_r_loss +=loss_r.item()
            #print(running_r_loss+1)
            running_t_loss+=loss_t.item()
        Train_Loss.append(running_loss / len(trainloader))

        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                BC = data["x"]
                Soln = data["y"]

                # forward + backward + optimize
                taskOut = tasknet(BC)
                ruleOut = rulenet(BC)
                alp = 0.5
                z = torch.cat((alp * taskOut, (1 - alp) * ruleOut), dim=-1)
                outputs = net(z).float()
                loss_t = criterion(outputs[:, 0, :], Soln[:, 0, :]) \
                + criterion(outputs[:, -1, :], Soln[:, -1, :]) + criterion(outputs[:, :, 0], Soln[:, :, 0])\
                         + criterion(outputs[:, :, -1], Soln[:, :, -1])

                loss_r = pdeloss2(outputs, args["lam"])
                loss = alp * loss_t + (1 - alp) * p * loss_r
                test_loss += loss.item()
        Test_Loss.append(test_loss / len(testloader))

        if Test_Loss[-1]<best_loss:
            best_loss = Test_Loss[-1]
            torch.save(net, 'tmp.pt')
            torch.save(tasknet,"tmp_tasknet.pt")
            torch.save(rulenet, "tmp_rulenet.pt")

        if verbose:
            print("Epoch: ", epoch)
            print("Train Loss: ", Train_Loss[-1])
            print("Test Loss: ", Test_Loss[-1])
            print("R loss: ",running_r_loss / len(trainloader) )
            print("T loss: ", running_t_loss / len(trainloader))
    combinedLoss = np.array([Train_Loss,Test_Loss])
    np.save(uniquify("./Models/Loss-CTRL.npy"),combinedLoss)
    os.rename('tmp.pt', uniquify('./Models/best-model-CTRL-CNN.pt'))
    os.rename('tmp_tasknet.pt', uniquify('./Models/best-model-CTRL-task.pt'))
    os.rename('tmp_rulenet.pt', uniquify('./Models/best-model-CTRL-rule.pt'))
    print('Finished Training')

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def visualize(vargs,args):
    lossName = vargs["lossName"]
    modelName = vargs["modelName"]
    loss = np.load(lossName)
    trainLoss = loss[0,:]
    testLoss = loss[1,:]
    epochs = np.linspace(0,len(trainLoss),len(trainLoss))
    plt.figure()
    plt.plot(epochs,trainLoss,label = "Train loss")
    plt.plot(epochs, testLoss, label= "Test loss")
    plt.legend()
    plt.title("Loss for heat equation with Deep CTRL")
    plt.ylabel("MSE loss")
    plt.xlabel("Epoch")
    plt.savefig(uniquify("./Figs/Loss.png"))
    plt.show()


    net = torch.load(modelName)
    rulenet = torch.load(vargs["rulenet"])
    tasknet = torch.load(vargs["tasknet"])
    #Save and load all models!
    Testset = PDE_Dataset(args["TestBC"], args["TestSoln"])
    SolnTens = Testset[0]["y"]
    x = np.linspace(-5, 5, SolnTens.shape[0])
    t = np.linspace(0, 3.14, SolnTens.shape[1])
    for i in range(1,5):

        BC = Testset[i+50]["x"]
        SolnTens = Testset[i+50]["y"]
        T, X = np.meshgrid(t, x)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        Y = SolnTens.cpu().detach().numpy()

        taskOut = tasknet(BC)
        ruleOut = rulenet(BC)
        alp = 0.5
        z = torch.cat((alp * taskOut, (1 - alp) * ruleOut), dim=-1)
        Yhat = torch.squeeze(net(z).float()).cpu().detach().numpy()
        ax.contour3D(X, T,Y , 50, cmap="winter")
        ax.contour3D(X, T, Yhat, 50, cmap="autumn")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('U(x,t)')
        plt.savefig(uniquify("./Figs/SolutionDeepCTRL.png"))
        plt.show()


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, T, (Y-Yhat)**2, 50, cmap="winter")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('Squared Residual')
        #plt.savefig(uniquify("./Figs/Solution.png"))
        print(np.sum((Y-Yhat)**2))
        plt.show()



def main():
    #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    args = {"TrainBC":"./Data/BC.npy","TrainSoln":"./Data/Soln.npy","TestBC": "./Data/BC_test.npy",
            "TestSoln":"./Data/Soln_test.npy","lam":6.28,"epochs":500,"device":device,"bz":batch_size}
    Trainset = PDE_Dataset(args["TrainBC"],args["TrainSoln"])
    Testset = PDE_Dataset(args["TestBC"],args["TestSoln"])
    trainloader = DataLoader(Trainset,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)

    s=(Trainset[0])["y"]
    M_x=s.shape[0]
    M_t= s.shape[1]
    net = Net(M_x,M_t)
    net.to(device)
    tn = taskNet(M_t)
    tn.to(device)
    rn = ruleNet(M_t)
    rn.to(device)
    #print(net)
    #print(tn)
    #print(rn)
    trainer(net,tn,rn,trainloader,testloader,args)
    vargs = {"rulenet":'./Models/best-model-CTRL-rule (11).pt',"tasknet": './Models/best-model-CTRL-task (11).pt'
        ,"modelName":"./Models/best-model-CTRL-CNN (11).pt","lossName": "./Models/Loss-CTRL (11).npy"}
    visualize(vargs,args)



#test()
main()