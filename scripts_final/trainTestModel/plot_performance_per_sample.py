import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
import pickle
import torch
import torch.optim as optim
import time

from Utilities.LossFunctionScript import *
from Utilities._loadModel import *
from Utilities.embed_data import *
from Utilities.plotScripts import *

config_file = 'config_files/foundParam_conv_4666_e3_100_small.yaml'

test_case_path = "/home/justinbrusche/datasets/foundationalParameters/case_12"


##############################
compute_diff_cfd = False
countCellweights = False
save_every_model = True
device = 'cuda'

with open(config_file, 'r') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

file_path = f"{test_case_path}/dataDictMeshes.pkl"
with open(file_path, 'rb') as file:
    dataDictMeshes = pickle.load(file)

#Save dataset -- Only has to be do once
# t1 = time.time()
# save_tensors(test_case_path, conf['nTrainData'] + conf['nTestData'], getPressure=True)
# get_faceArray_incl_boundaries(test_case_path)

# print(time.time() - t1)

BCInput: False
gradientBCInput: False

t1 = time.time()
# Load train / test data
# trainDataLoader = loadDataLoader(test_case_path, np.arange(conf['nTrainData']), conf['batchSize'],dataDictMeshes,BCInput=conf["modelParam"]["inputChannels"]["BCInput"],gradientBCInput=conf["modelParam"]["inputChannels"]["gradientBCInput"])
# testDataLoader = loadDataLoader(test_case_path_data, np.arange(1, 20),
#                                 1,dataDictMeshes,normalize_data=True)

testDataLoader = loadDataLoader(test_case_path, np.arange(0, 100,1),
                                1,dataDictMeshes,
                                normalize_data=True,
                                compute_diff_cfd=compute_diff_cfd,doShuffling=False)

# testDataLoader = loadDataLoader(test_case_path, np.arange(20, 70),
#                                 1,dataDictMeshes,
#                                 normalize_data=True,
#                                 compute_diff_cfd=False)

# testDataLoader = loadDataLoader(test_case_path_data, np.arange(0, 20),
#                                 1,dataDictMeshes,
#                                 normalize_data=True,
#                                 compute_diff_cfd=False)

# trainDataLoader = loadDataLoader(test_case_path, np.arange(conf['nTrainData']), 1,getPressure=True,pEqnD_isInput=conf["modelParam"]["inputChannels"]["pEqnD"])
# testDataLoader = loadDataLoader(test_case_path, np.arange(conf['nTrainData'], conf['nTrainData'] + conf['nTestData']),
#                                 1,getPressure=True,pEqnD_isInput=conf["modelParam"]["inputChannels"]["pEqnD"])

pointTypes_raw = None
if conf["modelParam"]["inputChannels"]["pointType"] == True:
    file_path = f"{test_case_path}/pointTypes.pt"
    pointTypes_raw = torch.load(file_path).float().to(device)
cellTypes_raw = None
SDF_raw = None
if conf["modelParam"]["inputChannels"]["cellType"] == True:
    file_path = f"{test_case_path}/cellTypes.pt"
    cellTypes_raw = torch.load(file_path).float()
# if conf["modelParam"]["inputChannels"]["SDF"] == True:
    file_path = f"{test_case_path}/SDF.pt"
    SDF_raw = torch.load(file_path).float()
    print("cellTypes_raw")

file_path = f"{test_case_path}/cellWeights.pt"
cellWeights = torch.load(file_path).float().to(device)
print(time.time() - t1)
# a

# Initialize model
sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")
# model = loadModel(conf,test_case_path_data,device)

file_path = f"{test_case_path}/embedding_conv_4666"
with open(file_path, 'rb') as file:
    GNNdataDict = pickle.load(file)

sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")
model = loadModel(conf,GNNdataDict["GNNData"],device)


# Load pretrained parameters
model_path = os.path.join(conf["modelDir"], f'{conf["modelFilename"]}.pth')
print(model_path)
if os.path.exists(model_path):
    print("Pre-trained model found. Loading model.")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    resume = True
else:
    print("No pre-trained model found. Initializing new model.")
    start_epoch = 0

    resume = False

file_path = f"{test_case_path}/embedding_{conf['edgeAttrName']}"
with open(file_path, 'rb') as file:
    GNNdataDict_pooling = pickle.load(file)

lossFunction = LossClass(test_case_path, conf,device)



# Function to run an epoch
def runEpoch(GNNdataDict,testDataLoader, model, lossFunction, epoch,dataDictMeshes,pointTypes_raw=None,cellTypes_raw=None):
    t = time.time()
    train_loss = 0
    lossSource_sum = 0
    lossPressure_sum = 0
    first_outP_test = None  # To store the first output

    loss_list = []
    # Evaluate on test data
    i = 0
    model.eval()
    test_loss = 0
    loss1_sum = 0
    loss2_sum = 0

    x_coords = dataDictMeshes[0]["cellCenters"][:, 0]
    y_coords = dataDictMeshes[0]["cellCenters"][:, 1]

    boundaryCells = dataDictMeshes[0]["boundaryCells"]

    x_coords_face = dataDictMeshes[0]["faceCoordinates_from_points"][:, 0]
    y_coords_face = dataDictMeshes[0]["faceCoordinates_from_points"][:, 1]

    cellPoints = dataDictMeshes[0]["cellPoints"]
    x_mesh = dataDictMeshes[0]["pointCoordinates"][:, 0]
    y_mesh = dataDictMeshes[0]["pointCoordinates"][:, 1]



    x_list = []
    y1_list = []
    y2_list = []
    y3_list = []
    y4_list = []
    i_sample = 0
    with torch.no_grad():
        for p, pEqnD, pEqnFace, pEqnSource, inputCellArray, faceArray_incl_boundaries, BCArray, *factors  in testDataLoader:
            print("i_sample", i_sample)
            i_sample += 1
            # print("factor",factor)
            # p = p / factor
            # print(inputCellArray.shape)
            # print(BCArray.shape)

            # print(dataDictMeshes[0]["faceCoordinates"][-len(BCArray[0]):])


            boundary_coordinates = dataDictMeshes[0]["faceCoordinates"][-len(BCArray[0]):]

            boundary_coordinates_trans = boundary_coordinates - np.mean(boundary_coordinates,axis=0)

            polar_angles = np.arctan2(boundary_coordinates_trans[:, 1], boundary_coordinates_trans[:, 0])

            # Get indices to sort based on polar angles
            sorted_indices = np.argsort(polar_angles)

            # print(boundary_coordinates_trans)
            # print(len(boundary_coordinates_trans))


            if compute_diff_cfd:
                inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 2))
                inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray
                inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)

            elif conf["modelParam"]["inputChannels"]["BCInput"] == True:
                if conf["modelParam"]["inputChannels"]["gradientBCInput"] == True:
                    inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 4))
                    inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                else:
                    inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 3))
                    inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)
            else:
                inputFaceArray = faceArray_incl_boundaries + 0

            if conf["modelParam"]["inputChannels"]["pointType"] == True:
                outP = model(inputCellArray.to(device), inputFaceArray.to(device) , GNNdataDict["graphData"], GNNdataDict["graphPoolData"],
                             GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"], pointTypes_raw)
            else:
                if conf["modelParam"]["inputChannels"]["cellType"] == True:
                    shape_input = inputCellArray.shape
                    cellTypes = cellTypes_raw.unsqueeze(0).unsqueeze(-1).expand(shape_input[0], shape_input[1], 1)
                    SDF = SDF_raw.unsqueeze(0).unsqueeze(-1).expand(shape_input[0], shape_input[1], 1)

                    inputCellArray = torch.cat([inputCellArray, cellTypes], dim=-1)
                    inputCellArray = torch.cat([inputCellArray, SDF], dim=-1)

                outP = model(inputCellArray.to(device), inputFaceArray.to(device), GNNdataDict["graphData"], GNNdataDict["graphPoolData"],
                             GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"])

            outP_copy = outP.to("cpu") * factors[3] * factors[2] + 0

            print(factors[3])

            # sp1, sp2, sp3, sp4, s1, s2, s3, s4, p4, pp4 = lossFunction.get_scale(outP, pEqnD, pEqnFace, pEqnSource, p)
            #
            # x_list.append(factors[3])
            # y1_list.append(torch.std(s1).cpu() / torch.std(sp1).cpu())
            # y2_list.append(torch.std(s2).cpu() / torch.std(sp2).cpu())
            # y3_list.append(torch.std(s3).cpu() / torch.std(sp3).cpu())
            # y4_list.append(torch.std(s4).cpu() / torch.std(sp4).cpu())

            if 1==1:
                outP = outP
                # sp1,sp2,sp3,sp4,s1,s2,s3,s4,p4,pp4 = lossFunction.get_scale(outP, pEqnD, pEqnFace, pEqnSource,p)
                #
                # x_list.append(factors[3])
                # y1_list.append(torch.std(s1).cpu()/torch.std(sp1).cpu())
                # y2_list.append(torch.std(s2).cpu()/torch.std(sp2).cpu())
                # y3_list.append(torch.std(s3).cpu()/torch.std(sp3).cpu())
                # y4_list.append(torch.std(s4).cpu()/torch.std(sp4).cpu())


                # lossSource = lossFunction.computeLossSource(outP, pEqnD.to(device), pEqnFace.to(device), pEqnSource.to(device),cellWeights,countCellweights)
                # #
                # loss = lossFunction.computeLossPressure(outP, p.to(device),cellWeights,countCellweights)
                #
                # loss = lossSource * 0.015 + loss
                # loss1_sum += lossSource.item()
                # loss_list.append(loss.item())


                lossPressure = lossFunction.computeLossPressure(outP, p.to(device))
                # lossSource = lossFunction.computeLossSource(outP,p.unsqueeze(-1).to(device), pEqnD.to(device), pEqnFace.to(device),
                #                                             pEqnSource.to(device), cellWeights, countCellweights,factors[3])
                # lossFVM = lossFunction.computeLossFVM(outP, p.to(device), pEqnFace.to(device))

                print("lossPressure, ",lossPressure.item())

                print(outP.shape)
                # outP = factors[4].unsqueeze(-1)
                # p = p.to("cpu") * factors[3] * factors[2] + factors[4]

                print(p)


                plot_type = '2d'

                zmin = min(np.min(p[0].numpy()), np.min(outP[0,:,0].cpu().numpy()), np.min(p[0].numpy() - outP[0,:,0].cpu().numpy()))
                zmax = max(np.max(p[0].numpy()), np.max(outP[0,:,0].cpu().numpy()), np.max(p[0].numpy() - outP[0,:,0].cpu().numpy()))
                zlim = [zmin, zmax]

                fig = plt.figure(figsize=(20, 11))
                if plot_type == '2d':
                    ax = fig.add_subplot(2, 2, 1)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0],"Ground_truth")
                    ax = fig.add_subplot(2,2, 2)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0,:,0].cpu(),"Prediction")
                    ax = fig.add_subplot(2,2, 3)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0,:,0].cpu()-p[0],"Diff",centered_colorbar=True)
                    ax = fig.add_subplot(2,2, 4)
                    plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, inputCellArray[0,:,0],"Source",centered_colorbar=True)
                    plt.tight_layout()

                    # ax = fig.add_subplot(2, 2, 1)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0],"Ground_truth",apply_percentile=True)
                    # ax = fig.add_subplot(2,2, 2)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0,:,0].cpu(),"Prediction",apply_percentile=True)
                    # ax = fig.add_subplot(2,2, 3)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0,:,0].cpu()-p[0],"Diff",centered_colorbar=True,apply_percentile=True)
                    # ax = fig.add_subplot(2,2, 4)
                    # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0],"Source",centered_colorbar=True,apply_percentile=True)
                    # plt.tight_layout()

                    # z = np.linspace(0, 5, len(boundary_coordinates_trans))

                    # Extract x and y from boundary_coordinates_trans
                    x = boundary_coordinates_trans[:, 0]
                    y = boundary_coordinates_trans[:, 1]

                    # Create 3D plot
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')

                    # Plot the points in 3D
                    ax.scatter(x, y, BCArray[0,:,2], c='b', marker='o', label='diri value')
                    ax.scatter(x, y, BCArray[0,:,0], c='r', marker='o', label='Neu')

                    ax.scatter(x, y, BCArray[0,:,1], c='g', marker='o', label='Diri')

                    ax.scatter(x, y, p[0,boundaryCells], c='k', marker='o', label='inner')

                    # Label axes
                    ax.set_xlabel('X Coordinate')
                    ax.set_ylabel('Y Coordinate')
                    ax.set_zlabel('Z Coordinate')
                    ax.set_title('3D Plot of Coordinates')

                    # Show legend
                    ax.legend()
                    #
                    #
                    #
                    # # fig, ax = plt.subplots(figsize=(10, 6))
                    # plt.figure()
                    # # plt.plot(BCArray[0,:,0][sorted_indices])
                    # # plt.plot(BCArray[0,:,1][sorted_indices])
                    # plt.plot(BCArray[0,:,2][sorted_indices],"o")
                    #
                    # plt.plot(p[0,boundaryCells][sorted_indices],'o')
                    #
                    #
                    # # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0,:,0].cpu()-p[0],"Pressure Field",centered_colorbar=True)
                    # plt.tight_layout()



                if plot_type == '3d':

                    ax = fig.add_subplot(1, 3, 1, projection='3d')
                    plot_3d(fig, ax, x_coords, y_coords, p[0], title="ground_truth", zlim=zlim)

                    ax = fig.add_subplot(1, 3, 2, projection='3d')
                    plot_3d(fig, ax, x_coords, y_coords, outP[0,:,0].cpu(), title="prediction", zlim=zlim)

                    ax = fig.add_subplot(1, 3, 3, projection='3d')
                    plot_3d(fig, ax, x_coords, y_coords, outP[0,:,0].cpu()-p[0], title="diff", zlim=zlim)

                    ax = fig.add_subplot(2,2, 4)
                    plot_3d(fig, ax, cellPoints, x_mesh, y_mesh, pEqnSource[0],"source")
                # plt.show()

                # print(a.shape)
                # plt.figure()
                # plt.scatter(a[0],torch.zeros(len(a[0])))
                # print(a[0])
                # print(factors[3],1/torch.std(sp1),1/torch.std(s4),1/torch.std(sp4))
                plt.show()

            elif 1==2:

                outP = outP.to("cpu") * factors[3] * factors[2] + factors[4].unsqueeze(-1)
                p = p.to("cpu") * factors[3] * factors[2] + factors[4]

                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(3, 2, 1)
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0], "ground_truth")
                ax = fig.add_subplot(3, 2, 2)
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0, :, 0].cpu(), "prediction")

                ax = fig.add_subplot(3, 2, 3)
                # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0]-outP[0,:,0].cpu(),"diff")
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0]-factors[4][0], "source")
                ax = fig.add_subplot(3, 2, 4)
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP[0, :, 0].cpu()-factors[4][0], "source prec")
                #
                print("aaaaaaaaaaa")
                print(torch.sum(abs(p[0]-factors[4][0])))
                print(torch.sum(abs(outP[0, :, 0].cpu()-factors[4][0])))
                ax = fig.add_subplot(3, 2, 5)
                # plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0]-outP[0,:,0].cpu(),"diff")
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, outP_copy, "pressure")
                ax = fig.add_subplot(3, 2, 6)
                plot_mesh_triangles(fig, ax, cellPoints, x_mesh, y_mesh, p[0]-outP[0, :, 0].cpu(), "pressure prec")

                plt.tight_layout()
                # plt.show()

                # print(a.shape)
                # plt.figure()
                # plt.scatter(a[0],torch.zeros(len(a[0])))
                # print(a[0])
                # print(factors[3],1/torch.std(sp1),1/torch.std(s4),1/torch.std(sp4))
                plt.show()

            # plt.tight_layout()
            # print(factors[3], 1 / torch.std(sp1), 1 / torch.std(s4), 1 / torch.std(sp4))
            # plt.show()

        plt.figure()
        # plt.scatter(x_list, y1_list)
        # plt.scatter(x_list, y2_list)
        # plt.scatter(x_list, y3_list)
        # plt.scatter(x_list, y4_list)
        # plt.plot(x_list,x_list)
        # plt.grid()
        # plt.show()

        # plt.figure()
        # plt.plot(factor_list_true,factor_list_guess,"o")
        # # Set equal limits for both axes
        plt.xlim(min(x_list + y1_list), max(x_list + y1_list))
        plt.ylim(min(x_list + y1_list), max(x_list + y1_list))
        x = np.linspace(0,max(x_list + y1_list),100)
        #
        plt.scatter(x_list,y1_list)
        plt.xlabel("true factor")
        plt.ylabel("guessed factor")
        plt.grid()
        plt.plot(x,0.9*x)
        plt.plot(x,1/0.9*x)
        plt.plot(x,x)

        #
        plt.show()



    avg_test_loss = test_loss / len(testDataLoader)


# Run multiple epochs and save the model intermittently
num_epochs = conf['maxEpochs']
save_interval = conf['freqToFile']

log_file_path = os.path.join(conf["modelDir"], 'training_log.pkl')

# Check if the log file already exists
if os.path.exists(log_file_path):
    with open(log_file_path, 'rb') as log_file:
        log_dict = pickle.load(log_file)
else:
    # Initialize the log dictionary if it doesn't exist
    log_dict = {"Epoch": [], "Train Loss": [], "Test Loss": []}
runEpoch(GNNdataDict,testDataLoader, model, lossFunction, 0,dataDictMeshes,pointTypes_raw=pointTypes_raw,cellTypes_raw=cellTypes_raw)

print("Training complete.")
