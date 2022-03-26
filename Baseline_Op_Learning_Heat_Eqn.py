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
    """Face Landmarks dataset."""

    def __init__(self, BC_file, Soln_file):
        """
        Args:
            BC_file (string): Path to .npy file with a Boundary condition tensor.
            Soln_file (string): Path to .npy file with a Grid solution tensor.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.SolnTens = torch.tensor(np.load(Soln_file)).float()
        self.BCTens = torch.tensor(np.load(BC_file)).float()

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
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5,5),padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5,5),padding=2)
        self.fc3 = nn.Linear(120,2* M_t * M_x*16)
        self.fc2 = nn.Linear(84,120)
        self.fc1 = nn.Linear(M_t,84)
        self.M_x = M_x
        self.M_t = M_t

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view((-1,2,self.M_x*4,self.M_t*4))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.squeeze(x,dim=1)

        return x

def trainer(net,trainloader,testloader,epochs,verbose=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    Train_Loss = []
    Test_Loss = []
    best_loss = 1000

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            BC = data["x"]
            Soln = data["y"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(BC).float()
            loss = criterion(outputs, Soln)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        Train_Loss.append(running_loss / len(trainloader))

        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                BC = data["x"]
                Soln = data["y"]

                # forward + backward + optimize
                outputs = net(BC).float()
                loss = criterion(outputs, Soln)
                test_loss += loss.item()
        Test_Loss.append(test_loss / len(testloader))

        if Test_Loss[-1]<best_loss:
            best_loss = Test_Loss[-1]
            torch.save(net, 'tmp.pt')

        if verbose:
            print("Epoch: ", epoch)
            print("Train Loss: ", Train_Loss[-1])
            print("Test Loss: ", Test_Loss[-1])
    combinedLoss = np.array([Train_Loss,Test_Loss])
    np.save(uniquify("./Models/Loss.npy"),combinedLoss)
    os.rename('tmp.pt', uniquify('./Models/best-model.pt'))
    print('Finished Training')

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def visualize(modelName,lossName,args):
    loss = np.load(lossName)
    trainLoss = loss[0,:]
    testLoss = loss[1,:]
    epochs = np.linspace(0,len(trainLoss),len(trainLoss))
    plt.figure()
    plt.plot(epochs,trainLoss,label = "Train loss")
    plt.plot(epochs, testLoss, label= "Test loss")
    plt.legend()
    plt.title("Loss for heat equation")
    plt.ylabel("MSE loss")
    plt.xlabel("Epoch")
    #plt.savefig(uniquify("./Figs/Loss.png"))
    #plt.show()


    model = torch.load(modelName)
    Testset = PDE_Dataset(args["TestBC"], args["TestSoln"])
    SolnTens = Testset[0]["y"]
    x = np.linspace(-5, 5, SolnTens.shape[0])
    t = np.linspace(0, 3.14, SolnTens.shape[1])
    for i in range(1,2):

        BC = Testset[i]["x"]
        SolnTens = Testset[i]["y"]
        T, X = np.meshgrid(t, x)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        Y = SolnTens.detach().numpy()
        Yhat = torch.squeeze(model(BC)).detach().numpy()
        ax.contour3D(X, T,Y , 50, cmap="winter")
        ax.contour3D(X, T, Yhat, 50, cmap="autumn")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('U(x,t)')
        plt.savefig(uniquify("./Figs/Solution.png"))
        plt.show()


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, T, Y-Yhat, 50, cmap="winter")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('Residual')
        plt.savefig(uniquify("./Figs/Solution.png"))
        plt.show()



def main():
    batch_size = 2
    args = {"TrainBC":"./Data/BC.npy","TrainSoln":"./Data/Soln.npy","TestBC": "./Data/BC_test.npy",
            "TestSoln":"./Data/Soln_test.npy"}
    Trainset = PDE_Dataset(args["TrainBC"],args["TrainSoln"])
    Testset = PDE_Dataset(args["TestBC"],args["TestSoln"])
    trainloader = DataLoader(Trainset,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)

    s=(Trainset[0])["y"]
    M_x=s.shape[0]
    M_t= s.shape[1]
    net = Net(M_x,M_t)
    epochs = 10
    #trainer(net,trainloader,testloader,epochs)
    visualize("./Models/best-model (2).pt","./Models/Loss (2).npy",args)



#test()
main()

